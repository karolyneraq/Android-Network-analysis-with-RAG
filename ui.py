# app.py
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import os
from os.path import join, dirname
from dotenv import load_dotenv
from langchain.chat_models import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

# ========================
# CONFIGURAÇÕES
# ========================

# Configure sua API key do Hugging Face (ou use variável de ambiente HF_API_KEY)
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
api_key = os.environ.get("GROQ_API_KEY")

# Inicialize o modelo Qwen via API da Groq
model = ChatGroq(model="groq/qwen-2.5-32b-instruct", api_key=api_key)


# ========================
# PIPELINE RAG
# ========================

embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def ingest_logs():
    logs = [
        {"origem": "1111-1111", "destino": "2222-2222", "data": "2025-09-20 14:32", "status": "atendida", "duracao": "3m20s"},
        {"origem": "3333-3333", "destino": "4444-4444", "data": "2025-09-21 18:12", "status": "perdida", "duracao": "0s"},
        {"origem": "5555-5555", "destino": "6666-6666", "data": "2025-09-22 09:45", "status": "atendida", "duracao": "12m10s"},
        {"origem": "7777-7777", "destino": "8888-8888", "data": "2025-09-23 21:30", "status": "rejeitada", "duracao": "0s"},
    ]
    return logs

def format_log(log: dict) -> str:
    return f"Chamada de {log['origem']} para {log['destino']} em {log['data']} - Status: {log['status']} - Duração: {log['duracao']}"

def index_logs(logs):
    texts = [format_log(log) for log in logs]
    embeddings = embedder.encode(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))
    return index, texts

def retrieve_relevant_logs(query: str, index, texts, top_k=3):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec, dtype=np.float32), top_k)
    return [texts[i] for i in I[0]]

def generate_response(query: str, retrieved_docs):
    """
    Gera resposta usando Qwen via API Groq com LangChain.
    """
    context = "\n".join(retrieved_docs)
    system_prompt = "Você é um assistente que analisa registros de chamadas telefônicas."
    user_prompt = f"""
        Use os registros abaixo para responder à pergunta do usuário de forma clara.

        Registros:
        {context}

        Pergunta: {query}
        Resposta:
        """
    # Cria mensagens para LangChain
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    # Chamada ao modelo Qwen via API Groq
    response = model(messages)
    return response.content

# ========================
# INTERFACE STREAMLIT
# ========================

st.set_page_config(page_title="Assistente de Logs Telefônicos", page_icon="📞", layout="centered")
st.title("📞 Assistente de Logs de Chamadas")
st.write("Pergunte sobre os registros de chamadas e o modelo Qwen responderá com base nos logs.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Olá! 👋 Sou o assistente de logs. Pergunte algo, por exemplo: *Quais chamadas não atendidas ontem?*"}
    ]

logs = ingest_logs()
index, texts = index_logs(logs)

# Mostrar histórico
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada do usuário
if prompt := st.chat_input("Digite sua pergunta sobre os logs..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    retrieved = retrieve_relevant_logs(prompt, index, texts)
    resposta = generate_response(prompt, retrieved)

    st.session_state["messages"].append({"role": "assistant", "content": resposta})
    with st.chat_message("assistant"):
        st.markdown(resposta)
