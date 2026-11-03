"""FinGuide - Streamlit chat UI. LangGraph RAG with streaming answers from Groq."""

import base64
import uuid
from pathlib import Path

import streamlit as st

import config
import rag_pipeline as rp

ASSETS = Path(__file__).parent / "assets"

st.set_page_config(page_title="FinGuide", page_icon="🏦")


def img_uri(name: str) -> str:
    """Return an SVG from assets/ as a data URI so it can be embedded in HTML."""
    data = base64.b64encode((ASSETS / name).read_bytes()).decode()
    return f"data:image/svg+xml;base64,{data}"


CSS = """<style>
.block-container{padding-top:1.5rem;max-width:850px;}
.fg-hero{display:flex;align-items:center;justify-content:space-between;gap:16px;background:linear-gradient(120deg,#064E3B 0%,#047857 55%,#10B981 100%);border-radius:20px;padding:20px 24px;color:#fff;box-shadow:0 8px 24px rgba(4,120,87,.25);margin-bottom:18px;}
.fg-hero-left{display:flex;align-items:center;gap:16px;}
.fg-logo{width:60px;height:60px;border-radius:14px;background:#fff;padding:4px;}
.fg-title{font-size:2rem;font-weight:800;line-height:1.1;letter-spacing:-.5px;}
.fg-tag{font-size:.92rem;opacity:.92;margin-top:4px;}
.fg-hero-right{display:flex;gap:10px;}
.fg-badge{background:rgba(255,255,255,.95);border-radius:14px;padding:8px;width:76px;height:76px;display:flex;align-items:center;justify-content:center;}
.fg-badge img{max-width:100%;max-height:100%;}
.fg-cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:8px 0 18px;}
.fg-card{background:#F1F8F4;border:1px solid #D1FAE5;border-radius:16px;padding:14px;text-align:center;}
.fg-card img{height:64px;margin-bottom:6px;}
.fg-card b{display:block;color:#065F46;margin-bottom:2px;}
.fg-card span{font-size:.85rem;color:#4B5B52;}
.fg-stats{display:flex;gap:8px;flex-wrap:wrap;margin:4px 0 10px;}
.fg-pill{background:#ECFDF5;color:#065F46;border:1px solid #A7F3D0;border-radius:999px;padding:3px 10px;font-size:.78rem;font-weight:600;}
.fg-note{font-size:.8rem;color:#6B7A72;text-align:center;margin-top:6px;}
@media (max-width:640px){.fg-hero-right{display:none;}.fg-cards{grid-template-columns:1fr;}}
</style>"""

st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    "<div class='fg-hero'><div class='fg-hero-left'>"
    f"<img class='fg-logo' src='{img_uri('logo.svg')}'/>"
    "<div><div class='fg-title'>FinGuide</div>"
    "<div class='fg-tag'>Personal finance literacy assistant · based on RBI and SEBI booklets · "
    "LangChain + LangGraph · powered by Groq</div></div></div>"
    "<div class='fg-hero-right'>"
    f"<div class='fg-badge'><img src='{img_uri('bank.svg')}'/></div>"
    f"<div class='fg-badge'><img src='{img_uri('cash.svg')}'/></div>"
    "</div></div>",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Preparing the knowledge base (the first run reads the PDFs and downloads the "
                                "embedding model, so it can take a few minutes)...")
def get_pipeline() -> rp.RAGPipeline:
    return rp.RAGPipeline()


try:
    pipeline = get_pipeline()
except Exception as exc:  # embedding model download or document loading problem
    st.error(f"Could not start the knowledge base: {exc}\n\n"
             "Check your internet connection (needed once to download the embedding model) "
             "and that the `knowledge_base/` folder contains readable documents.")
    st.stop()


def show_sources(sources: list[rp.Source]) -> None:
    if not sources:
        return
    with st.expander("📚 Sources"):
        st.caption("Page numbers refer to the page in the PDF viewer.")
        for s in sources:
            badge = " · 📎 your upload" if s.origin == "your upload" else ""
            st.markdown(f"**{s.filename}** · similarity {s.score:.2f}{badge}")
            st.caption(s.excerpt)


def error_message(exc: Exception, model: str) -> str:
    if isinstance(exc, rp.MissingAPIKeyError):
        return ("**No Groq API key found.**\n\nPaste your own key in the sidebar, or (for the app owner) add "
                "`GROQ_API_KEY` to the Streamlit secrets or a `.env` file. Free keys are available at console.groq.com.")
    if isinstance(exc, rp.LLMAuthError):
        return "**Groq rejected the API key.** Check that the key is correct and still active, then try again."
    if isinstance(exc, rp.RateLimitedError):
        return ("**Groq's rate limit was reached.** Wait a minute and try again, pick a smaller model such as "
                "`openai/gpt-oss-20b`, or use your own API key in the sidebar.")
    if isinstance(exc, rp.ModelNotFoundError):
        return (f"**Model `{model}` is not available on Groq.** Groq retires models from time to time. "
                "Choose another model in the sidebar or check console.groq.com/docs/models.")
    if isinstance(exc, rp.LLMUnavailableError):
        return "**Could not reach Groq.** Check your internet connection or Groq's status page, then try again."
    return f"**Groq returned an error:** {exc}"


@st.cache_data(ttl=900, show_spinner=False)
def fetch_models(api_key: str | None) -> list[str]:
    return rp.list_groq_models(api_key)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = uuid.uuid4().hex   # LangGraph chat-memory thread for this session

# ---------------------------------------------------------------- sidebar --
with st.sidebar:
    st.markdown(f"<img src='{img_uri('logo.svg')}' width='56'/>", unsafe_allow_html=True)
    st.subheader("Settings")
    server_key = config.get_secret("GROQ_API_KEY")
    user_key = st.text_input(
        "Your Groq API key (optional)" if server_key else "Groq API key",
        type="password",
        placeholder="paste-your-groq-key-here",
        help="Kept only in this browser session. Get a free key at console.groq.com. "
             "With your own key there are no per-session question limits.",
    ).strip()
    api_key = user_key or server_key
    live_models = fetch_models(api_key)
    options = list(live_models) or list(config.FALLBACK_MODELS)
    if config.DEFAULT_MODEL not in options:
        options.insert(0, config.DEFAULT_MODEL)
    model = st.selectbox("Model (Groq)", options, index=options.index(config.DEFAULT_MODEL),
                         help="Groq retires models from time to time, so this list is loaded live when a key is available.")
    if not api_key:
        st.warning("No API key yet. Paste your Groq key above to start chatting.")

    st.subheader("Knowledge base")
    st.markdown(
        f"<div class='fg-stats'><span class='fg-pill'>📄 {len(pipeline.documents)} documents</span>"
        f"<span class='fg-pill'>🧩 {pipeline.base_chunk_count} passages</span></div>",
        unsafe_allow_html=True,
    )
    with st.expander("See loaded documents"):
        for name in pipeline.documents:
            st.caption(f"• {name}")
        st.caption("Front matter, contact lists, blank worksheets and personalised advice letters are not indexed.")
    for w in pipeline.warnings:
        st.warning(w)

    uploads = st.file_uploader(
        "Add your own documents (this session only)",
        type=[e.lstrip(".") for e in sorted(rp.SUPPORTED_EXTENSIONS)],
        accept_multiple_files=True,
        help="PDF, Word (.docx), TXT or Markdown. Answers can then use them as sources. "
             "To keep documents permanently, put them in the knowledge_base/ folder.",
    )
    if uploads:
        signature = tuple((f.name, f.size) for f in uploads)
        if st.session_state.get("upload_sig") != signature:
            with st.spinner("Reading your documents..."):
                store, count, notes = pipeline.add_uploads([(f.name, f.getvalue()) for f in uploads])
            st.session_state.user_store = store
            st.session_state.upload_sig = signature
            st.session_state.upload_notes = notes
            st.session_state.upload_count = count
        for note in st.session_state.get("upload_notes", []):
            st.warning(note)
        if st.session_state.get("user_store"):
            st.success(f"Added {st.session_state.upload_count} passages from your uploads.")
    else:
        st.session_state.user_store = None
        st.session_state.upload_sig = None

    st.subheader("Try asking")
    examples = [
        "What is the Rule of 72?",
        "What are the steps of financial planning?",
        "What is envelope budgeting?",
        "What is the National Pension System (NPS)?",
        "How can I avoid Ponzi schemes?",
        "How do I file a complaint on SCORES?",
    ]
    clicked = None
    for q in examples:
        if st.button(q, use_container_width=True):
            clicked = q
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = uuid.uuid4().hex   # forget the conversation memory too

prompt = st.chat_input("Ask a personal finance question...") or clicked

# ---------------------------------------------------------------- welcome --
if not st.session_state.messages and not prompt:
    st.markdown(
        "<div class='fg-cards'>"
        f"<div class='fg-card'><img src='{img_uri('cash.svg')}'/><b>Money Management</b>"
        "<span>Saving, record keeping, budgeting, goals</span></div>"
        f"<div class='fg-card'><img src='{img_uri('bank.svg')}'/><b>Banking, Loans &amp; Insurance</b>"
        "<span>Accounts, digital banking, credit, debt, cover</span></div>"
        f"<div class='fg-card'><img src='{img_uri('coins.svg')}'/><b>Investing &amp; Safety</b>"
        "<span>Markets, mutual funds, NPS, Ponzi schemes</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.info("Ask a question below or pick one from the sidebar. FinGuide remembers the conversation, so you can "
            "ask follow-ups. It gives general education only, not financial advice.")

# ------------------------------------------------------------------- chat --
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🏦" if msg["role"] == "assistant" else "🧑"):
        st.markdown(msg["content"])
        show_sources(msg.get("sources", []))

limited = False
if prompt:
    using_shared_key = not user_key
    if len(prompt) > config.MAX_QUESTION_CHARS and using_shared_key:
        st.warning(f"Please keep questions under {config.MAX_QUESTION_CHARS} characters, or use your own Groq key.")
        prompt, limited = None, True
    elif using_shared_key and st.session_state.get("question_count", 0) >= config.MAX_QUESTIONS_PER_SESSION:
        st.warning(f"This shared demo allows {config.MAX_QUESTIONS_PER_SESSION} questions per session. "
                   "Paste your own free Groq key in the sidebar to continue.")
        prompt, limited = None, True

if prompt:
    st.session_state.question_count = st.session_state.get("question_count", 0) + 1
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🏦"):
        try:
            run = pipeline.ask(prompt, st.session_state.thread_id, model, st.session_state.get("user_store"), api_key)
            stream = run.tokens()
            with st.spinner("Searching the knowledge base..."):
                first = next(stream, None)          # condense -> retrieve -> grade happen before the first token

            def tokens():
                if first is not None:
                    yield first
                yield from stream

            answer = st.write_stream(tokens())      # tokens appear as the model generates them
            if run.standalone_question.strip() != prompt.strip():
                st.caption(f"🔎 Searched for: {run.standalone_question}")
            sources = run.sources if run.relevant else []
            show_sources(sources)
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        except rp.LLM_ERRORS as exc:
            text = error_message(exc, model)
            st.error(text)
            st.session_state.messages.append({"role": "assistant", "content": text})

st.markdown("<div class='fg-note'>General educational information only. Not financial advice. "
            "Your questions and the retrieved passages are sent to Groq to generate answers.</div>",
            unsafe_allow_html=True)
