# 📄 Ask My Docs – RAG PDF Q&A App

## 📌 Project Description

Ask My Docs is a simple AI-powered web application that allows users to upload PDF documents and ask questions in natural language. The system retrieves the most relevant sections from the uploaded PDFs and generates accurate answers using a Retrieval-Augmented Generation (RAG) pipeline.

Instead of manually reading long documents, users can simply ask questions and get direct answers with proper document-based references.

---

## 🚀 Features

- Upload multiple PDF files (minimum 3 supported)
- Automatic text chunking (1000 characters with overlap)
- SentenceTransformer embeddings (MiniLM model)
- Chroma vector database for semantic search
- Top-3 relevant chunk retrieval
- AI-powered answers using Groq LLM
- Source-based responses from uploaded PDFs
- Simple and clean Streamlit interface

---

## 🧠 How It Works

- PDFs are uploaded by the user  
- Text is extracted and split into smaller chunks  
- Each chunk is converted into embeddings  
- Embeddings are stored in ChromaDB  
- User asks a question  
- System retrieves top relevant chunks  
- Groq LLM generates final answer using only retrieved context  
- Sources are displayed for transparency  

---

## 🛠️ Technologies Used

- Python 3.10+
- Streamlit
- LangChain
- ChromaDB
- SentenceTransformers (all-MiniLM-L6-v2)
- Groq API
- PyPDF

---

## 📂 Project Structure

```bash
ask-my-docs/
│
├── app.py
├── requirements.txt
├── chroma_db/
├── screenshots/
│   ├── upload.png
│   ├── answer.png
│   └── sources.png
└── README.md
Question ans Answer Output
Q1.What is this pdf about?
This document is the Microsoft Public License (Ms-PL), which governs the use of accompanying software and outlines the terms and conditions of its use, distribution, and modification.
Q2.what is thie key content in it?
This document outlines the Microsoft Public License (Ms-PL) which grants users a non-exclusive, royalty-free copyright and patent license to use, distribute and modify the accompanying software.
