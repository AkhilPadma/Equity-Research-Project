import os
import streamlit as st
import time

from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS

# Import Mistral AI support
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

# Load environment variables (MISTRAL_API_KEY must be set in .env)
load_dotenv()

st.title("Equity Research Tool ")
st.sidebar.title("News Article URLs")

# Collect up to 3 URLs from the sidebar
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

# Define a dedicated data folder for FAISS storage
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# FAISS saves as a folder, not .pkl
file_path = os.path.join(DATA_DIR, "faiss_store_mistral")

main_placeholder = st.empty()

# Initialize Mistral LLM
llm = ChatMistralAI(
    model="mistral-medium",  # Options: mistral-tiny, mistral-small, mistral-medium
    temperature=0.9,
    max_tokens=500,
    mistral_api_key=os.getenv("MISTRAL_API_KEY")
)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...")
    data = loader.load()

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...")
    docs = text_splitter.split_documents(data)

    # create embeddings with Mistral
    embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    mistral_api_key=os.getenv("MISTRAL_API_KEY")
    )

    vectorstore_mistral = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...")
    time.sleep(2)

    # Save FAISS index safely (creates a folder under data/)
    vectorstore_mistral.save_local(file_path)

# Input query
# Input query
query = main_placeholder.text_input("Question: ")

if query:
    if os.path.exists(file_path):
        embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=os.getenv("MISTRAL_API_KEY")
        )
        # Load FAISS index safely
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

        # Display answer
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
