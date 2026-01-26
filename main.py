import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

load_dotenv()

st.set_page_config(page_title="Equity Research Tool", layout="wide")
st.title("Equity Research Tool")
st.sidebar.title("News Article URLs")

MISTRAL_API_KEY = None
try:
    MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY", None)
except Exception:
    pass
if not MISTRAL_API_KEY:
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    st.error("Missing MISTRAL_API_KEY. Add it in Streamlit Cloud → App → Settings → Secrets.")
    st.stop()

# Collect up to 3 URLs
urls = []
for i in range(3):
    u = st.sidebar.text_input(f"URL {i+1}").strip()
    if u:
        urls.append(u)

process_url_clicked = st.sidebar.button("Process URLs")

# Storage path for FAISS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_DIR = os.path.join(DATA_DIR, "faiss_store_mistral")

main_placeholder = st.empty()

# Mistral LLM
llm = ChatMistralAI(
    model="mistral-medium",
    temperature=0.2,
    max_tokens=600,
    mistral_api_key=MISTRAL_API_KEY,
)

def build_embeddings():
    return MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=MISTRAL_API_KEY,
    )

if process_url_clicked:
    if not urls:
        st.warning("Enter at least one valid URL.")
    else:
        main_placeholder.info("Loading articles...")
        loader = WebBaseLoader(web_paths=urls)  
        data = loader.load()

        main_placeholder.info("Splitting text...")
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ".", ","],
            chunk_size=1000,
            chunk_overlap=150,
        )
        docs = splitter.split_documents(data)

        main_placeholder.info("Creating embeddings + building FAISS index...")
        embeddings = build_embeddings()
        vs = FAISS.from_documents(docs, embeddings)

        vs.save_local(INDEX_DIR)
        main_placeholder.success("Done. Ask a question below.")
        time.sleep(0.5)

st.divider()
query = st.text_input("Question:")

if query:
    if not os.path.exists(INDEX_DIR):
        st.error("FAISS index not found. Click 'Process URLs' first.")
        st.stop()

    embeddings = build_embeddings()
    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    )

    with st.spinner("Thinking..."):
        result = chain.invoke({"question": query})

    st.header("Answer")
    st.write(result.get("answer", ""))

    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources")
        for s in sources.split("\n"):
            s = s.strip()
            if s:
                st.write(s)
