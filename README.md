# Wikipedia-AI

# From Text to Meaning: Knowledge-Based Embeddings and RAG

A project exploring **Retrieval-Augmented Generation (RAG)** using knowledge embeddings, LangChain, and GPT-3.5, turning raw Wikipedia text into a context-aware, intelligent Q&A system.

---

## Overview

This project demonstrates how to supercharge a Large Language Model (LLM) with external, factual knowledge using RAG. Instead of relying solely on what GPT-3.5 was trained on, this system retrieves relevant information from a custom knowledge base at query time, giving the model long-term, accurate, and domain-specific context.

---

## How It Works

```
User Question
     │
     ▼
[Embedding Model] ──► Query Vector
     │
     ▼
[Vector Search] ──► Relevant Snippets (from Wikipedia embeddings)
     │
     ▼
[GPT-3.5 via LangChain] ──► Contextual, Accurate Response
```

1. **Vectorize the Knowledge** -> Wikipedia's AI-related pages are chunked and converted into numerical embeddings.
2. **Semantic Search** -> When a user asks a question, the query is embedded and matched against the knowledge base to retrieve the most relevant snippets.
3. **Augmented Generation** -> The retrieved snippets are passed alongside the question to GPT-3.5, which generates a grounded, context-aware response.

---

## Tech Stack

| Component | Tool |
|---|---|
| Orchestration | [LangChain](https://www.langchain.com/) |
| LLM | OpenAI GPT-3.5 |
| Embeddings | OpenAI Embeddings API |
| Vector Database | [FAISS](https://faiss.ai/) |
| Knowledge Source | Wikipedia (AI-related pages) |

---

## Key Learnings

- **Text → Embeddings**: How to represent natural language as dense numerical vectors for efficient similarity search.
- **Semantic Retrieval**: Building a search pipeline that finds contextually relevant passages rather than relying on keyword matching.
- **RAG Architecture**: How retrieval and generation work together to make LLM outputs more accurate and factual.
- **LangChain Chains**: Composing retrieval and generation steps into a clean, reusable pipeline.

---

## Getting Started

### Prerequisites

```bash
pip install langchain openai
```

### Environment Variables

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Running the Project

```bash
python main.py
```
