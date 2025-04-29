# Local AI Search Engine

This project is a **local file search engine** developed using **Python**, **Streamlit**, and the **OLLAMA Llama3.2:3B** AI model.  
It enables intelligent file searches based on natural language queries.

---

## Project Overview

- **Indexing**: Python scans and indexes the content of local files.
- **AI Search**: The **Llama3.2:3B** model (via Ollama) processes user queries and identifies relevant files by searching indexed content.
- **Streamlit UI**: A clean and simple web interface allows users to interact with the search engine.
- **Citation Mapping**: The system matches citation indexes from the AI output to corresponding files and returns the most relevant results.

---

## How It Works

1. **File Indexing**:  
   Python reads file contents and creates an index for efficient searching.

2. **User Query**:  
   Through the **Streamlit** interface, the user submits a natural language search query.

3. **AI Response**:  
   The query is sent to the **OLLAMA** model (`llama3.2:3b`), which searches through indexed data and returns citations.

4. **Result Mapping**:  
   Using citation indexes, the system retrieves and displays the matching files.

---

## Technology Stack

- **Python 3.9** (slim build via Docker)
- **Streamlit** for the user interface
- **Ollama Llama3.2:3B** for AI-powered search
- **Docker** for containerization

---

## Requirements

- Docker installed
- Ollama installed and running with the `llama3.2:3b` model available

---

## Setup and Running the Application

### 1. Clone the Repository

```bash
git clone https://github.com/gokulindhu/local-search-engine.git
cd local-search-engine
