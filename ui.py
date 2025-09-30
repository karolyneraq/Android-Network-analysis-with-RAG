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

def generate_response(user_text, log_text, retrieved_pairs):
    examples = ""
    for p in retrieved_pairs:
        examples += f"Log:\n{p['log']}\nAnalysis:\n{p['analysis']}\n\n"

    system_prompt = (
        "Você é um especialista em análise de logs SIP (mensagens SIPMSG). /no_think "
        "Sempre responda de forma clara, direta e em português brasileiro. "
        "Evite explicações internas ou raciocínios 'para dentro'."
    )

    # Monta o prompt do usuário de acordo com o que foi enviado
    if user_text and log_text:
        user_prompt = f"""
        O usuário enviou a seguinte mensagem: 
        "{user_text}"

        Ele também anexou este log para análise:
        {log_text}

        Exemplos de logs e análises para contexto:
        {examples}

        Agora, faça a análise combinando a mensagem e o log do usuário.
        """
    elif log_text:
        user_prompt = f"""
        O usuário não escreveu uma mensagem, mas enviou o seguinte log:

        {log_text}

        Exemplos de logs e análises para contexto:
        {examples}

        Agora, faça a análise desse log.
        """
    else:
        user_prompt = f"""
        O usuário enviou a seguinte pergunta/mensagem: 
        "{user_text}"

        Exemplos de logs e análises para contexto:
        {examples}

        Agora, responda à mensagem do usuário de forma objetiva.
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

st.set_page_config(page_title="Assistente de Logs SIP", page_icon="📞", layout="wide")

st.title("Assistente de Logs SIP")
st.write("Você pode digitar perguntas e/ou enviar logs para análise.")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Olá! Sou o assistente de logs SIP. Pergunte algo ou envie logs para análise."}
    ]

# Indexa logs existentes
pairs = ingest_folder("database")
index, stored_pairs = index_pairs(pairs)

# ========================
# Histórico do chat
# ========================
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ========================
# Uploader de log (anexa mas não envia sozinho)
# ========================
uploaded_file = st.file_uploader("📎 Anexar log (.txt)", type="txt")

if "pending_log" not in st.session_state:
    st.session_state["pending_log"] = None

if uploaded_file is not None:
    st.session_state["pending_log"] = uploaded_file.read().decode("utf-8").strip()
    st.success("📎 Log anexado! Ele será enviado junto com a próxima mensagem.")

# ========================
# Função de envio de mensagem
# ========================
def process_input(user_text, log_text=None):
    # Exibe a mensagem no histórico
    msg_display = user_text if user_text else ""
    if log_text:
        msg_display += f"\n\n📎 Arquivo enviado:\n{log_text[:500]}..."  # mostra só um trecho

    st.session_state["messages"].append({"role": "user", "content": msg_display})
    with st.chat_message("user"):
        st.markdown(msg_display)

    # Prepara entrada pro modelo
    query_for_model = user_text if user_text else ""
    if log_text:
        query_for_model += "\n\n(Log completo anexado)"

    retrieved = retrieve_relevant_pairs(query_for_model, index, stored_pairs)
    response = generate_response(user_text, log_text, retrieved)

    st.session_state["messages"].append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# ========================
# Caixa de input fixo (chat_input nativo do Streamlit)
# ========================
user_text = st.chat_input("Digite sua pergunta ou comentário...")

if user_text:
    log_text = st.session_state["pending_log"]
    process_input(user_text, log_text)

    # Limpa o log pendente (não reenvia em todas as mensagens)
    st.session_state["pending_log"] = None
