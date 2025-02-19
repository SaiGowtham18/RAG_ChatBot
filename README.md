# RAG Chatbot

## 📌 Project Overview
This project implements a **Retrieval-Augmented Generation (RAG) chatbot** capable of answering user queries by retrieving relevant information from PDF files and other unstructured text sources. The chatbot integrates **FAISS vector search**, **LLM-based response generation**, and **prompt optimization** to deliver accurate, contextual answers.

## 🚀 Key Features
- **Contextual Information Retrieval**: Uses **FAISS** for efficient semantic search over embedded text data.
- **Dynamic RAG Pipeline**: Enhances LLM responses by fetching relevant content before query execution.
- **Optimized Token Usage**: Reduces costs and improves response efficiency via **prompt engineering**.
- **Streamlit-Based Interactive UI**: Enables users to upload PDFs, ask questions, and receive accurate responses.

## 🛠 Tech Stack
- **Programming**: Python  
- **Machine Learning & NLP**: OpenAI embeddings, FAISS, LangChain  
- **Frameworks & Libraries**: OpenAI API, GPT-3.5, SentenceTransformers, PyTorch  
- **Vector Search**: FAISS  
- **Deployment & UI**: Streamlit  

## 📌 Approach
1. **Data Preprocessing**: Extracts and chunks text from uploaded PDFs.  
2. **Embedding Generation**: Converts text into vector representations using **OpenAI embeddings**.  
3. **FAISS-Based Search**: Retrieves relevant text chunks from the database based on semantic similarity.  
4. **LLM Response Generation**: Uses **ChatGPT-3.5** with retrieved context for enhanced answer accuracy.  
5. **Token Optimization**: Applies **prompt engineering** to minimize cost and improve performance.  

