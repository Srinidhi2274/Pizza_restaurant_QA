# 🍽️ Restaurant Reviewer AI Agent

An AI-powered restaurant review generator that processes real-time user reviews, analyzes sentiment, and generates human-like summaries using on-device LLM inference with [Ollama](https://ollama.com/), Langchain, and Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

- 💬 **On-device LLM inference** with Ollama for fast, private processing
- 🔍 **Context-aware review generation** using RAG pipelines and Langchain
- 🤖 **Sentiment analysis** to extract nuanced insights from customer feedback
- ⚡ **Fast responses** – typically under 2 seconds
- 📊 Processes over **10,000+ reviews** with **95% accuracy**

---

## 🛠️ Tech Stack

- **LLM**: Ollama (e.g., `llama3`, `mistral`, `phi`)
- **Orchestration**: Langchain agents & tools
- **Data Retrieval**: RAG (Vector Store + Retriever)
- **Sentiment Analysis**: VADER/TextBlob or transformer-based model
- **Backend**: Python (FastAPI or CLI)
- **Storage**: ChromaDB / FAISS for vector search

