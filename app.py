from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
import faiss
import markdown
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
# FLASK APP
# ========================
app = Flask(__name__)

# ========================
# RAG FUNCTIONS
# ========================
def ingest_folder(folder_path="database"):
    pairs = []
    if not os.path.exists(folder_path):
        return pairs
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

pairs = ingest_folder("database")
texts = [p["log"] for p in pairs]
if texts:
    embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)
else:
    faiss_index, pairs = None, []

def retrieve_relevant_pairs(query, faiss_index, pairs, k=3):
    if faiss_index is None or not pairs or not query:
        return []
    query_vec = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = faiss_index.search(query_vec, k)
    return [pairs[i] for i in I[0]]

def generate_response(user_text, log_text, retrieved_pairs):
    """
    user_text: optional string (user typed message)
    log_text: optional string (uploaded log content)
    retrieved_pairs: list of {log,analysis,metadata} from RAG
    """
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
        \"{user_text}\"

        Ele também anexou este log para análise:
        {log_text}

        Exemplos de logs e análises para contexto:
        {examples}

        Agora, faça a análise combinando a mensagem e o log do usuário.
        """
    # Caso o usuário peça análise mas não envie log
    elif (user_text and "analise" in user_text.lower()) and not log_text:
        return (
            "Você pediu uma análise de log, mas nenhum arquivo foi enviado. "
            "Por favor, anexe o log (.txt) para que eu possa analisá-lo corretamente."
        )

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
        \"{user_text}\"

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
# ROUTES
# ========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form.get("user_input", "").strip()
    retrieved = retrieve_relevant_pairs(user_input, faiss_index, pairs)
    response_text = generate_response(user_input, None, retrieved)

    # converte Markdown para HTML
    response_html = markdown.markdown(response_text)

    return f"""
    <div class="user"><b>Você:</b> {user_input}</div>
    <div class="assistant"><b>Assistente:</b> {response_html}</div>
    """



@app.route("/upload", methods=["POST"])
def upload():
    user_input = request.form.get("user_input", "").strip()
    file = request.files.get("file")
    if not file and not user_input:
        return "<div class='error'>Nenhum arquivo enviado e nenhuma mensagem fornecida.</div>"

    content = ""
    filename = ""
    if file:
        filename = file.filename
        content = file.read().decode("utf-8").strip()

    query_for_retrieval = content if content else user_input
    retrieved = retrieve_relevant_pairs(query_for_retrieval, faiss_index, pairs)
    response_text = generate_response(user_input if user_input else None,
                                      content if content else None,
                                      retrieved)

    # converte Markdown para HTML
    response_html = markdown.markdown(response_text)

    user_html = f"<div class='user'><b>Você:</b> {user_input}</div>" if user_input else ""
    file_html = f"<div class='user'><b>Arquivo:</b> {filename}</div>" if filename else ""

    return f"""
    {user_html}
    {file_html}
    <div class="assistant"><b>Assistente:</b> {response_html}</div>
    """

from sklearn.metrics.pairwise import cosine_similarity

def evaluate_response(user_text, log_text, expected_response):
    # Recupera os documentos relevantes (RAG)
    retrieved = retrieve_relevant_pairs(user_text, faiss_index, pairs)
    
    # Gera a resposta do modelo
    generated = generate_response(user_text, log_text, retrieved)
    
    # Calcula embeddings das respostas
    emb_expected = embedder.encode([expected_response], convert_to_numpy=True, normalize_embeddings=True)
    emb_generated = embedder.encode([generated], convert_to_numpy=True, normalize_embeddings=True)
    
    # Calcula similaridade de cosseno
    similarity = cosine_similarity(emb_expected, emb_generated)[0][0]
    print("=================================================================\n\n\n\n")
    print("=====   Usuário:", user_text, "  =====")
    print("=====   Esperado:   =====\n", expected_response)
    print("=====   Gerado:   =====\n", generated)
    print(f"=====  Similaridade de cosseno: {similarity:.3f}  =====\n")
    
    return similarity

# Exemplo de uso:
test_user_text = "Analise o log de chamada SIP anexado."
test_log_text = f"""
09-29 13:28:55.220  4258  4704 I SIPMSG[0,2]: [<--] INVITE sip:724xxxxxxxxx477@[2804:0214:930C:7033:1869:CD61:10A3:044C]:5060 SIP/2.0 [CSeq: 975900 INVITE]
09-29 13:28:55.339  4258  4704 I SIPMSG[0,2]: [-->] SIP/2.0 180 Ringing [CSeq: 975900 INVITE]
09-29 13:29:25.645  4258  4704 I SIPMSG[0,2]: [-->] SIP/2.0 486 Busy Here [CSeq: 975900 INVITE]
09-29 13:29:25.699  4258  4704 I SIPMSG[0,2]: [<--] CANCEL sip:724xxxxxxxxx477@[2804:0214:930C:7033:1869:CD61:10A3:044C]:5060 SIP/2.0 [CSeq: 975900 CANCEL]
09-29 13:29:25.701  4258  4704 I SIPMSG[0,0]: [-->] SIP/2.0 481 Call/Transaction Does Not Exist [CSeq: 975900 CANCEL]
"""
expected_response = f"""
[<--] INVITE ... -> O UE recebeu uma chamada.
[-->] 180 Ringing -> O telefone começou a tocar, informando que o destino está sendo alertado.
[-->] 486 Busy Here -> O UE envia Busy Here, indicando que não pode atender (usuário ocupado).
[<--] CANCEL ... -> O servidor tenta cancelar a chamada, mas a UE já havia respondido com 486.
[-->] 481 Call/Transaction Does Not Exist -> UE confirma que a transação já não existe, ou seja, a chamada já foi encerrada.
"""

evaluate_response(test_user_text, test_log_text, expected_response)


if __name__ == "__main__":
    app.run(debug=True)
