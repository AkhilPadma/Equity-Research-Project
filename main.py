import os
import time
import streamlit as st
import requests
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

load_dotenv()

st.set_page_config(page_title="Equity Research Tool", layout="wide")
st.title("Equity Research Tool")
st.sidebar.title("News Article URLs")

# --- Get API key (Secrets first, env second) ---
MISTRAL_API_KEY = None
try:
    MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY", None)
except Exception:
    pass
if not MISTRAL_API_KEY:
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    st.error("Missing MISTRAL_API_KEY. Add it in Streamlit Cloud → Manage app → Settings → Secrets.")
    st.stop()

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {MISTRAL_API_KEY}", "Content-Type": "application/json"}

# --- URLs ---
urls = []
for i in range(3):
    u = st.sidebar.text_input(f"URL {i+1}").strip()
    if u:
        urls.append(u)

process_url_clicked = st.sidebar.button("Process URLs")

# --- Local index dir ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_DIR = os.path.join(DATA_DIR, "faiss_store")

main_placeholder = st.empty()

# --- Embeddings: simple + local (no extra SDK) ---
# Use LangChain's built-in FakeEmbeddings? No.
# Better: use Mistral embeddings via HTTP so we don't depend on langchain-mistralai.

def mistral_embed(texts):
    # Mistral embeddings endpoint (requires model that supports embeddings)
    # If your account supports "mistral-embed", this works.
    r = requests.post(
        "https://api.mistral.ai/v1/embeddings",
        headers=HEADERS,
        json={"model": "mistral-embed", "input": texts},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()
    return [item["embedding"] for item in data["data"]]

class MistralEmbeddings:
    def embed_documents(self, texts):
        return mistral_embed(texts)

    def embed_query(self, text):
        return mistral_embed([text])[0]

embeddings = MistralEmbeddings()

if process_url_clicked:
    if not urls:
        st.warning("Enter at least one valid URL.")
    else:
        main_placeholder.info("Loading articles...")
        loader = WebBaseLoader(web_paths=urls)
        docs = loader.load()

        main_placeholder.info("Splitting text...")
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","],
            chunk_size=800,
            chunk_overlap=120,
        )
        chunks = splitter.split_documents(docs)

        main_placeholder.info("Building FAISS index...")
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(INDEX_DIR)

        main_placeholder.success("Done. Ask a question below.")
        time.sleep(0.5)

st.divider()
query = st.text_input("Question:")

if query:
    if not os.path.exists(INDEX_DIR):
        st.error("Index not found. Click 'Process URLs' first.")
        st.stop()

    vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

    with st.spinner("Retrieving sources..."):
        retrieved = vs.similarity_search(query, k=4)

    context = "\n\n---\n\n".join([d.page_content for d in retrieved])
    sources = [d.metadata.get("source", "") for d in retrieved]

    prompt = f"""You are an equity research assistant.
Use ONLY the context below to answer the question. If the answer is not in the context, say "Not found in provided articles."

CONTEXT:
{context}

QUESTION:
{query}

Answer concisely, then list sources (URLs) if available.
"""

    with st.spinner("Thinking..."):
        resp = requests.post(
            MISTRAL_API_URL,
            headers=HEADERS,
            json={
                "model": "mistral-medium",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
            },
            timeout=90,
        )

    if resp.status_code != 200:
        st.error(resp.text)
        st.stop()

    answer = resp.json()["choices"][0]["message"]["content"]

    st.header("Answer")
    st.write(answer)

    clean_sources = [s for s in sources if s]
    if clean_sources:
        st.subheader("Sources")
        for s in dict.fromkeys(clean_sources):
            st.write(s)
