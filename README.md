# 🔍 LLM-Powered Financial Analyst — A RAG System for Earnings Call Q&A

This project is an **industry-grade Retrieval-Augmented Generation (RAG)** pipeline built to intelligently answer user questions from **company earnings call transcripts**. It handles both **document-level** and **topic-level** questions by classifying them using an LLM agent, retrieves relevant chunks using multiple search strategies, and generates accurate, faithful answers.

---

## 🚀 Features

- 📄 **Handles both topic-level and document-level questions**
  - Classifies questions using an LLM-based agent
  - Generates answers either by summarizing the full document or retrieving relevant chunks

- 🔍 **Multiple retrieval strategies**
  - Supports **MMR**, **exact match**, and **hybrid** search
  - Compatible with **OpenAI**, **BGE**, and other embedding models

- 🧠 **Modular and extensible RAG pipeline**
  - Built with **LangChain**, supports plug-and-play chains and prompts
  - Clean code structure and logging using Python best practices

- 📈 **Robust evaluation framework**
  - Uses **RAGAS** to evaluate metrics like **faithfulness**, **context recall**, and **answer relevance**

---

## 📂 Project Structure

rag-financial-qa/
├── ingestion/ # Data loading and cleaning
│ └── load_and_clean_pdf.py
│ └── split_docs.py
├── prompts/ # Prompt templates for different tasks
│ └── question_classifier_prompt.py
│ └── response_prompt.py
│ └── doc_summary.py
├── retriever/ # Retrieval and RAG chain integration
│ └── create_retriever.py
│ └── integrate_chains.py
├── indexing/ # Embedding and vector store setup (e.g., Pinecone)
│ └── create_index.py
├── evaluation/ # RAG evaluation using RAGAS
│ └── evaluate_rag.py
├── main.py # Main script to run the pipeline
├── requirements.txt
└── README.md

## 📊 Evaluation with RAGAS

The project includes an evaluation pipeline using [RAGAS](https://github.com/explodinggradients/ragas) to assess the quality of generated answers based on key metrics:

- **Context Recall**: Checks whether relevant context was retrieved from the document.
- **Context Precision**: measures the signal-to-noise ratio of the retrieved context
- **Faithfulness**: Measures if the answer is grounded in the retrieved context.
- **Answer Relevance**: Evaluates how relevant the generated answer is to the user question.

### 🧱 Tech Stack (Markdown Format)
- **LangChain** for orchestration

- **Pinecone** as vector store

- **OpenAI / Hugging Face** LLMs and embedding models

- **RAGAS** for evaluation

- Python **logging, modular pipeline**, unit testing

Reference- https://medium.com/data-science/evaluating-rag-applications-with-ragas-81d67b0ee31a
