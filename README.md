# Bang! Rules RAG Pipeline

This project is a RAG pipeline specialized for the card game Bang!. It indexes the local Bang rule sources from `BangRules/`, retrieves the most relevant rule chunks, generates answers through the configured OpenAI API, and evaluates against FAQ data from `FAQ/`.

![rag-image](./rag-design-basic.png)

## Overview

- Indexes Bang rules from LaTeX source files in `BangRules/`
- Stores embeddings in LanceDB
- Uses local reranking instead of Cohere
- Generates rule-grounded answers about Bang!
- Loads evaluation questions from `FAQ/*.pdf` or from a JSON file

## Architecture

- `main.py`
  Entry point for reset, add, query, evaluate, and run commands.
- `src/impl/indexer.py`
  Extracts Bang card descriptions, glossary items, and rule lists from the LaTeX rulebook.
- `src/impl/datastore.py`
  Stores embeddings in LanceDB.
- `src/impl/retriever.py`
  Combines vector retrieval with a local lexical reranking pass.
- `src/impl/response_generator.py`
  Produces Bang-specific answers from retrieved rule chunks.
- `src/util/faq_loader.py`
  Extracts evaluation question-answer pairs from FAQ PDFs.

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Set your OpenAI key before running the pipeline:

```powershell
$env:OPENAI_API_KEY="your_openai_api_key"
```

## Usage

Default paths:

```python
DEFAULT_SOURCE_PATH = "BangRules"
DEFAULT_EVAL_PATH = "FAQ"
```

Run the full pipeline:

```bash
python main.py run
```

Reset the vector database:

```bash
python main.py reset
```

Index Bang rules:

```bash
python main.py add -p "BangRules"
```

Ask a question:

```bash
python main.py query "Ako funguje karta Pivo pri poslednych dvoch hracoch?"
```

Run the Streamlit chat UI:

```bash
streamlit run streamlit_app.py
```

The app uses the existing index as-is. Reindexing is available only from developer mode.

For deployment, you can protect the app with passwords through Streamlit secrets or environment variables:

```toml
APP_PASSWORD = "your-user-password"
DEV_PASSWORD = "your-developer-password"
OPENAI_API_KEY = "your-openai-api-key"
```

- `APP_PASSWORD` protects the chatbot for normal users
- `DEV_PASSWORD` unlocks developer mode
- `OPENAI_API_KEY` is read from Streamlit secrets first and from environment variables as fallback
- if `DEV_PASSWORD` is missing, developer mode is available without a second password
- if `APP_PASSWORD` is missing, the app opens without a login gate

Evaluate against FAQ:

```bash
python main.py evaluate -f "FAQ"
```
