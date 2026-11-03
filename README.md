# FinGuide: Personal Finance Literacy Assistant

A RAG chatbot built with **LangChain** and **LangGraph**, powered by **Groq** for fast hosted inference, with a public-ready **Streamlit** interface. Its knowledge base is two official Indian financial-education books: the **SEBI Financial Education Booklet** and the **RBI "ICanDo Financial Planning"** book. Answers stream word by word, and the app can run locally or on **Streamlit Community Cloud**.

**Note:** FinGuide gives general educational information, not financial advice.

## Architecture

**Ingestion (once, then cached in `.cache/`)**

```
PDF/DOCX/TXT loaders (LangChain) -> drop unwanted pages -> remove headers/footers
  -> RecursiveCharacterTextSplitter -> local embeddings (MiniLM, via FastEmbed) -> FAISS index
```

**Question answering: a LangGraph state graph with chat memory**

```
START -> condense -> retrieve -> grade --relevant-------> generate -> END
                                    \--not relevant--> fallback -> END
```

| Node | What it does |
|---|---|
| `condense` | Rewrites a follow-up such as "explain that more simply" into a standalone question using the chat history. It only calls the model when the question looks like a follow-up. |
| `retrieve` | Finds the top passages in the knowledge base (and in any documents you uploaded) with FAISS cosine similarity. |
| `grade` | Relevance gate: if the best similarity is below the threshold, the question is treated as out of scope. |
| `generate` | `ChatPromptTemplate -> ChatGroq -> StrOutputParser`, streamed token by token, with a system prompt that forbids personal advice and invented rates. |
| `fallback` | Polite "not in my knowledge base" reply; the answer model is not called. |

LangGraph's `MemorySaver` keeps the conversation per chat thread ("Clear chat" starts a new one). Only the language model runs on Groq; documents and embeddings stay on the app's own server.

## Knowledge base

| File | What it is |
|---|---|
| `Financial_Education_Booklet_-_English.pdf` | SEBI, Nov 2020, 73 pages: key concepts, financial planning, savings, securities market, insurance, pension, borrowing, government schemes, tax saving, Ponzi schemes, grievance redressal. |
| `I_Can_Do_RBI.pdf` | RBI "ICanDo Financial Planning", 2010, 143 pages: money management, budgeting, goals, investing, debt, risk and insurance. |

**Not indexed** (see `SKIP_PAGES` in `config.py`, PDF page numbers): disclaimers and contents pages, contact lists, the story and foreword, blank worksheets, the book list, and the personalised "Dear Ravi / Dear Gita..." advice letters (they conflict with the no-personal-advice rule). Running headers and page numbers are removed. About 560 passages are indexed.

**Keep in mind**
- The books date from 2020 and 2010. Rates, limits and tax rules have changed; FinGuide is told to point users to official sources for current figures.
- Page numbers shown in sources are PDF viewer pages.
- **Copyright:** the SEBI booklet's notice forbids reproduction without written permission, and RBI allows reproduction with acknowledgement. A public app shows short excerpts from these PDFs and a public repo contains them. For a public deployment, get permission from SEBI, or remove the SEBI PDF, and always cite the sources. A private repository avoids publishing the files, but the app still displays excerpts.

## Features

- Streaming answers, chat memory and follow-up questions.
- Sources (file, PDF page, excerpt) under every answer, and "Searched for: ..." when a follow-up was rewritten.
- Model picker loaded live from Groq, with a fallback list.
- Session-only uploads of PDF, Word, TXT and Markdown files, searched together with the books.
- Visitors can paste their own Groq key. Without one, a shared demo allows `MAX_QUESTIONS_PER_SESSION` questions of up to `MAX_QUESTION_CHARS` characters each, to protect your key.
- Index cache: the first start reads the PDFs and embeds them (a minute or two); later starts load in about a second.
- Clear messages for missing/invalid keys, rate limits, retired models and network problems.

## Models (Groq)

| Model | Notes |
|---|---|
| `openai/gpt-oss-120b` (default) | Best quality of the production models; reasoning model. |
| `openai/gpt-oss-20b` | Faster and cheaper, a good fallback when you hit limits. |

Groq retires models regularly. For example, `llama-3.3-70b-versatile` and `llama-3.1-8b-instant` were retired on 16 Aug 2026 for free and developer tiers. The sidebar lists the models your key can currently use; check https://console.groq.com/docs/models. Set a different default with `GROQ_MODEL`. For gpt-oss models FinGuide uses low reasoning effort to keep answers quick.

## Run locally

1. Get a free API key at https://console.groq.com/keys.
2. Use Python 3.11 or 3.12, then in the `FinGuide` folder:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
   (In PowerShell, if activation is blocked: `Set-ExecutionPolicy -Scope Process RemoteSigned`.)
3. Add your key, either way:
   - copy `.env.example` to `.env` and set `GROQ_API_KEY`, or
   - copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set `GROQ_API_KEY`.
   You can also skip this and paste the key into the sidebar.
4. Run `streamlit run app.py`.
5. The first start downloads the small embedding model (about 90 MB) and reads the PDFs, so wait for the loading message to finish.

## Deploy on Streamlit Community Cloud (public UI)

1. **Put the project on GitHub** (all files, including `knowledge_base/`). Make sure `.env` and `.streamlit/secrets.toml` are **not** committed (they are in `.gitignore`). Mind the copyright note above before making the repo public.
2. Go to https://share.streamlit.io, sign in with GitHub, and choose **Create app**.
3. Select your repository and branch, and set the main file to `app.py`.
4. Open **Advanced settings**: choose **Python 3.12**, and paste your secrets:
   ```
   GROQ_API_KEY = "paste-your-groq-key-here"
   GROQ_MODEL = "openai/gpt-oss-120b"
   ```
5. Click **Deploy**. The first start builds the index (a few minutes). After that the app is live at a public `*.streamlit.app` link.

**Good to know for a public app**
- It runs on **your** Groq key and quota. The per-session limits help, but a popular app can still hit Groq's rate limits; visitors can use their own key. Watch usage in the Groq console and rotate the key if it leaks.
- Visitors' questions and the retrieved passages are sent to Groq. The footer says so; consider adding a privacy note if you share the link widely.
- Apps go to sleep after a period without visitors and rebuild the index when they wake. To make wake-ups fast, run the app once locally, remove `.cache/` from `.gitignore`, and commit the folder `.cache/index_*` (about 2 MB). It is reused only if the PDFs, chunk settings and embedding model are unchanged.
- Community Cloud has modest memory and CPU. That is why embeddings use the light FastEmbed (ONNX) backend instead of PyTorch. If you prefer sentence-transformers locally: `pip install langchain-huggingface sentence-transformers` and set `EMBEDDING_BACKEND=huggingface`.

## Adding documents

- **Permanently:** put `.pdf`, `.docx`, `.txt` or `.md` files in `knowledge_base/` and restart (the index rebuilds automatically).
- **For one session:** use "Add your own documents" in the sidebar (marked 📎 in sources).
- Scanned PDFs, password-protected files and the old `.doc` format are skipped with a warning. To hide pages of a new PDF, add its file name and page ranges to `SKIP_PAGES` in `config.py`.

## Configuration

Set these as environment variables, in `.env`, or in Streamlit secrets:

| Setting | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | none | Groq API key |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | Default model |
| `MIN_RELEVANCE_SCORE` | `0.25` | Relevance gate (lower = fewer fallback replies) |
| `MAX_QUESTIONS_PER_SESSION` | `25` | Limit for visitors using your shared key |
| `MAX_QUESTION_CHARS` | `600` | Maximum question length on the shared key |
| `EMBEDDING_BACKEND` | `fastembed` | `fastembed` or `huggingface` |

Chunk size, `top_k`, temperature and token limits are in `config.py`.

## Troubleshooting

- **"No Groq API key found":** add `GROQ_API_KEY` (secrets, `.env`) or paste a key in the sidebar.
- **"Groq rejected the API key":** the key is wrong, expired or revoked; create a new one.
- **"Rate limit reached":** wait a minute, choose `openai/gpt-oss-20b`, or use another key.
- **"Model not available":** Groq retired it. Pick another from the sidebar list.
- **Too many "not in my knowledge base" replies:** lower `MIN_RELEVANCE_SCORE` (for example `0.2`).
- **Slow first start:** it is embedding the PDFs once. Delete `.cache/` to force a rebuild.
- **Follow-ups feel slow:** rewriting a follow-up adds one short model call. Set `CONDENSE_FOLLOWUPS = False` in `config.py` to switch it off.
- **Install problems:** use 64-bit Python 3.11/3.12 and keep the version ranges in `requirements.txt`; LangChain and LangGraph change quickly.
- **Cloud app crashes on start:** open "Manage app" -> logs. Memory errors usually mean too many extra documents; keep uploads small.

## Project structure

```
FinGuide/
├── app.py                      # Streamlit UI (streaming chat, uploads, model + key settings)
├── rag_pipeline.py             # LangChain ingestion + LangGraph question-answering graph + Groq
├── config.py                   # settings, secrets lookup, page filters (SKIP_PAGES)
├── assets/                     # logo.svg, bank.svg, cash.svg, coins.svg
├── knowledge_base/
│   ├── Financial_Education_Booklet_-_English.pdf   # SEBI
│   └── I_Can_Do_RBI.pdf                            # RBI
├── .streamlit/
│   ├── config.toml             # theme colors
│   └── secrets.toml.example    # template for your Groq key
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
