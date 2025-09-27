This is a research tool built with Streamlit, LangChain, FAISS, and Mistral AI. It allows users to input news article URLs, process their content, and ask natural language questions. The system retrieves relevant information from the processed articles and provides answers with cited sources.

This project demonstrates how to combine document loading, text splitting, embeddings, vector search, and a large language model (LLM) to create a question-answering application.

Features

Accepts multiple news article URLs as input.

Extracts and cleans text from the articles using UnstructuredURLLoader.

Splits long documents into smaller chunks for better embedding and retrieval.

Converts text chunks into embeddings using MistralAIEmbeddings.

Stores embeddings in a FAISS vector database for efficient similarity search.

Uses ChatMistralAI as the language model to answer user queries.

Returns answers along with references to the original sources.

Interactive web application built with Streamlit

<img width="1918" height="914" alt="image" src="https://github.com/user-attachments/assets/004f6a2c-0901-4346-a414-f8163315e096" />
