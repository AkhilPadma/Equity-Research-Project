import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

load_dotenv()

st.title("Equity Research Tool")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    u = st.sidebar.text_input(f"URL {i+1}").strip()
    if u:
        urls.append(u)

process_url_clicked = st.sidebar.button("Process URLs")

API_KEY = st.secrets.get("MISTRAL_API_KEY") or os.getenv("MISTRAL_API_KEY")
if not API_KEY:
    st.error("Missing MISTRAL_API_KEY. Add it in Streamlit Secrets or env vars.")
    st.stop()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
file_path = os.path.join(DATA_DIR, "faiss_store_mistral")

main_placeholder = st.empty()

llm = ChatMistralAI(
    model="mistral-medium",
    temperature=0.9,
    max_tokens=500,
    mistral_api_key=API_KEY
)

if process_url_clicked:
    if not urls:
        st.sidebar.error("Please enter at least one valid URL.")
        st.stop()

    main_placeholder.text("Data Loading...Started...")
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    main_placeholder.text("Text Splitter...Started...")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    docs = text_splitter.split_documents(data)

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=API_KEY
    )

    main_placeholder.text("Building FAISS index...")
    vectorstore_mistral = FAISS.from_documents(docs, embeddings)
    vectorstore_mistral.save_local(file_path)
    main_placeholder.success("Index built and saved.")

query = st.text_input("Question:")

if query:
    if os.path.exists(file_path):
        embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=API_KEY
        )

        vectorstore = FAISS.load_local(
            file_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        result = chain({"question": query}, return_only_outputs=True)

        st.header("Answer")
        st.write(result.get("answer", ""))

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
    else:
        st.warning("No index found. Process URLs first.")
