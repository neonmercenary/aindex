# aindex (Agentic Index) 🧠📄

**aindex** is a local-first, privacy-focused agentic research assistant designed to index, search, and reason over large local document libraries. Built with a lightweight, production-grade tech stack, it combines traditional relational database capabilities with blazing-fast vector similarity search—all running locally on your hardware without heavy framework bloat or cloud dependencies.

---

## 🚀 Key Features

* **Local-First & Privacy-Centric:** All documents, embeddings, and LLM inferences stay entirely on your machine.
* **Hybrid Relational + Vector Storage:** Powered by **SQLite** and the `sqlite-vec` extension, allowing you to run standard SQL queries and KNN vector searches within a single file database.
* **Zero PyTorch Overhead:** Uses **`fastembed`** (running on ONNX Runtime) to generate local embeddings quickly and efficiently on standard CPUs.
* **Multimodal Extraction:** Leverages **`pdfplumber`** to extract both standard narrative text and structured data tables page-by-page.
* **Custom LLM Integration:** Connects seamlessly with local LLMs via **Ollama** (supporting massive context windows and custom GGUF models with strict page-citation system prompts).
* **Modern Chat Dashboard:** A clean, dark-themed HTML/JS frontend featuring a unified chat interface, instant PDF drop-zone uploads, and quick-action research chips.

---

## 🛠️ Tech Stack

* **Backend Framework:** FastAPI (Python 3.11+)
* **Database & Vector Search:** SQLite + `sqlite-vec` (`vec0` virtual tables)
* **Embeddings:** `fastembed` (`BAAI/bge-small-en-v1.5`)
* **PDF Parsing:** `pdfplumber`
* **Inference Engine:** Ollama (Local GGUF models)
* **Data Validation:** Pydantic v2 & `pydantic-settings`
* **Package Management:** `uv`

---

## 📂 Project Structure

```text
aindex/
├── app/
│   ├── __init__.py
│   ├── db.py          # Database initialization, connections, and vector serialization[cite: 1]
│   ├── main.py        # FastAPI app, routing, upload, and RAG generation logic[cite: 2]
│   ├── schemas.py     # Pydantic request/response models[cite: 3]
│   └── settings.py    # Environment configuration[cite: 4]
├── static/
│   └── index.html     # Modern chat dashboard frontend
├── pyproject.toml
└── README.md