import os
import time
import streamlit as st
import requests
from dotenv import load_dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

load_dotenv()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Equity Research Tool", layout="wide")
st.title("Equity Research Tool")
st.sidebar.title("News Article URLs")

# -----------------------------
# Secrets / API key
# -----------------------------
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

MISTRAL_CHAT_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_EMBED_URL = "https://api.mistral.ai/v1/embeddings"

HEADERS = {
    "Authorization": f"Bearer {MISTRAL_API_KEY}",
    "Content-Type": "application/json",
}

# -----------------------------
# Inputs
# -----------------------------
urls = []
for i in range(3):
    u = st.sidebar.text_input(f"URL {i+1}").strip()
    if u:
        urls.append(u)

process_url_clicked = st.sidebar.button("Process URLs (Rebuild Index)")

# -----------------------------
# Local index path (optional persistence)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_DIR = os.path.join(DATA_DIR, "faiss_store")

main_placeholder = st.empty()

# -----------------------------
# Mistral embeddings via HTTP (no langchain-mistralai dependency)
# -----------------------------
def mistral_embed(texts):
    r = requests.post(
        MISTRAL_EMBED_URL,
        headers=HEADERS,
        json={"model": "mistral-embed", "input": texts},
        timeout=90,
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

# -----------------------------
# Build / Load index helpers
# -----------------------------
def build_index(urls_list):
    """Build FAISS index from URLs, store in session, and save locally."""
    loader = WebBaseLoader(web_paths=urls_list)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=800,
        chunk_overlap=120,
    )
    chunks = splitter.split_documents(docs)

    vs_local = FAISS.from_documents(chunks, embeddings)
    vs_local.save_local(INDEX_DIR)

    st.session_state["vs"] = vs_local
    st.session_state["indexed_urls"] = list(urls_list)
    st.session_state["vs_ready"] = True

def load_index_if_available():
    """Load FAISS index from disk if it exists (for fresh app restarts)."""
    if "vs" in st.session_state:
        return st.session_state["vs"]

    if os.path.exists(INDEX_DIR):
        vs_local = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        st.session_state["vs"] = vs_local
        st.session_state["vs_ready"] = True
        return vs_local

    return None

# -----------------------------
# Process URLs button behavior
# -----------------------------
if process_url_clicked:
    if not urls:
        st.warning("Enter at least one valid URL.")
    else:
        try:
            main_placeholder.info("Loading articles & building index...")
            build_index(urls)
            main_placeholder.success("Index built successfully. Ask a question below.")
            time.sleep(0.5)
        except Exception as e:
            st.exception(e)
            st.stop()

# -----------------------------
# Q&A
# -----------------------------
st.divider()
query = st.text_input("Question:")

if query:
    # Auto-build if missing (so user never hits "Index not found")
    vs = load_index_if_available()

    if vs is None:
        if not urls:
            st.error("No index found and no URLs provided. Enter URLs first (left sidebar).")
            st.stop()
        try:
            with st.spinner("Index missing. Building it now..."):
                build_index(urls)
            vs = st.session_state["vs"]
        except Exception as e:
            st.exception(e)
            st.stop()

    # Retrieve context
    with st.spinner("Retrieving sources..."):
        retrieved = vs.similarity_search(query, k=4)

    context = "\n\n---\n\n".join([d.page_content for d in retrieved])
    sources = [d.metadata.get("source", "") for d in retrieved]
    sources = [s for s in sources if s]
    sources = list(dict.fromkeys(sources))  # unique, preserve order

    prompt = f"""You are an equity research assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say: "Not found in provided articles."

CONTEXT:
{context}

QUESTION:
{query}

Return:
1) Answer (concise)
2) Sources (URLs) on new lines
"""

    # Call Mistral chat
    try:
        with st.spinner("Thinking..."):
            resp = requests.post(
                MISTRAL_CHAT_URL,
                headers=HEADERS,
                json={
                    "model": "mistral-medium",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.2,
                },
                timeout=120,
            )
        if resp.status_code != 200:
            st.error(resp.text)
            st.stop()

        answer = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.exception(e)
        st.stop()

    st.header("Answer")
    st.write(answer)

    if sources:
        st.subheader("Sources")
        for s in sources:
            st.write(s)
