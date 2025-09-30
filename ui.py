import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from os.path import join, dirname
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage

# ========================
# CONFIG
# ========================

dotenv_path = join(dirname(__file__), ".env")
load_dotenv(dotenv_path)
api_key = os.environ.get("GROQ_API_KEY")

model = ChatGroq(model="qwen/qwen3-32b", api_key=api_key)
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ========================
# INGESTION
# ========================

def ingest_folder(folder_path="database"):
    pairs = []
    for fname in os.listdir(folder_path):
        if fname.startswith("log") and fname.endswith(".txt"):
            suffix = fname[len("log-"):-4]
            analysis_fname = f"analysis-{suffix}.txt"
            analysis_path = join(folder_path, analysis_fname)
            if not os.path.exists(analysis_path):
                continue
            log_path = join(folder_path, fname)
            with open(log_path, "r", encoding="utf-8") as f:
                log_text = f.read().strip()
            with open(analysis_path, "r", encoding="utf-8") as f:
                analysis_text = f.read().strip()
            pairs.append({
                "log": log_text,
                "analysis": analysis_text,
                "metadata": {"source": fname}
            })
    return pairs

def index_pairs(pairs):
    texts = [p["log"] for p in pairs]
    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = embeddings.astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, pairs

def retrieve_relevant_pairs(query, index, pairs, k=3):
    query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(query_vec, k)
    return [pairs[i] for i in I[0]]

# ========================
# RESPONSE GENERATION
# ========================

def generate_response(query, retrieved_pairs):
    examples = ""
    for p in retrieved_pairs:
        examples += f"Log:\n{p['log']}\nAnalysis:\n{p['analysis']}\n\n"

    system_prompt = (
        "Você é um especialista em análise de logs SIP (mensagens SIPMSG). /no_think "
        "Sempre responda de forma clara, direta e em português brasileiro. "
        "Evite explicações internas ou raciocínios 'para dentro'."
    )
    user_prompt = f"""
    Aqui estão exemplos de logs e suas análises:

    {examples}

    Agora, responda à seguinte pergunta ou analise o seguinte log do usuário:

    {query}

    Responda de forma objetiva, clara e concisa, usando português brasileiro.
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = model(messages)
    return response.content

# ========================
# STREAMLIT UI
# ========================

st.set_page_config(page_title="Assistente de Logs SIP", page_icon="📞", layout="centered")
st.title("Assistente de Logs SIP")
st.write("Você pode digitar perguntas ou enviar logs para análise.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Olá! Sou o assistente de logs SIP. Pergunte algo ou envie logs para análise."}
    ]

if "uploaded_file_processed" not in st.session_state:
    st.session_state["uploaded_file_processed"] = False

# Indexa logs existentes
pairs = ingest_folder("database")
index, stored_pairs = index_pairs(pairs)

# Mostra histórico
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Função de envio de mensagem
def process_input(text):
    st.session_state["messages"].append({"role": "user", "content": text})
    with st.chat_message("user"):
        st.markdown(text)

    retrieved = retrieve_relevant_pairs(text, index, stored_pairs)
    response = generate_response(text, retrieved)

    st.session_state["messages"].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# ----------------------
# Ordem garantida: chat_input → botão de upload
# ----------------------

user_input = st.chat_input("Digite uma pergunta ou cole novos logs...")



if user_input:
    process_input(user_input)
st.title("teste")
uploaded_file = st.file_uploader("Ou envie um log .txt para análise", type="txt", key="upload_botao")
if uploaded_file is not None and not st.session_state["uploaded_file_processed"]:
    log_text = uploaded_file.read().decode("utf-8").strip()
    process_input(log_text)
    st.session_state["uploaded_file_processed"] = True
