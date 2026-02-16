# --- SQLite fix for Chroma (MUST be first) ---
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

# --- Normal imports ---
import re
from pathlib import Path
import chromadb
import streamlit as st
from bs4 import BeautifulSoup
from chromadb.utils import embedding_functions
from openai import OpenAI

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
HTML_DIR = BASE_DIR / "HW4-HTML"
DB_DIR = BASE_DIR / "chroma_hw4_db"

# ----------------------------
# HTML -> text
# ----------------------------
def html_to_text(html_path: Path) -> str:
    raw = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------------
# Chunking: EXACTLY TWO chunks per doc
# Method: split near midpoint (simple + deterministic)
# Why: meets HW requirement; keeps both "front half" + "back half" of each doc searchable
# ----------------------------
def two_chunks(text: str) -> list[str]:
    if not text:
        return ["", ""]

    mid = len(text) // 2

    # Try to split on a sentence-ish boundary near mid for cleaner chunks
    window = 400
    left = max(0, mid - window)
    right = min(len(text), mid + window)
    slice_ = text[left:right]

    # Find best split point in that window
    candidates = [m.start() for m in re.finditer(r"[\.!?]\s", slice_)]
    if candidates:
        split_local = candidates[len(candidates)//2]
        split_idx = left + split_local + 1
    else:
        split_idx = mid

    return [text[:split_idx].strip(), text[split_idx:].strip()]

# ----------------------------
# Build / load vector DB once
# ----------------------------
def get_collection():
    chroma_client = chromadb.PersistentClient(path=str(DB_DIR))

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=st.secrets["OPEN_API_KEY"],
        model_name="text-embedding-3-small",
    )

    collection = chroma_client.get_or_create_collection(
        name="HW4Collection",
        embedding_function=openai_ef,
    )
    return collection

def ingest_html_if_needed(collection):
    if collection.count() > 0:
        return

    html_files = sorted(HTML_DIR.glob("*.html"))
    if not html_files:
        st.error(f"No HTML files found in {HTML_DIR}")
        st.stop()

    ids, docs, metas = [], [], []
    for f in html_files:
        text = html_to_text(f)
        parts = two_chunks(text)

        for i, part in enumerate(parts, start=1):
            ids.append(f"{f.name}::part{i}")
            docs.append(part)
            metas.append({"source": f.name, "part": i})

    collection.add(ids=ids, documents=docs, metadatas=metas)

def retrieve_context(collection, query: str, k: int = 6):
    res = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas"],
    )
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    sources = sorted({m.get("source", "") for m in metas if m})

    context = "\n\n---\n\n".join(docs)
    return context, sources

def keep_last_5_interactions(messages):
    # 1 system + last 10 non-system messages (5 user/assistant pairs)
    system = [m for m in messages if m["role"] == "system"][:1]
    rest = [m for m in messages if m["role"] != "system"]
    return system + rest[-10:]

# ----------------------------
# Streamlit App
# ----------------------------
st.title("HW4: Student Orgs Chatbot (RAG)")

if "client" not in st.session_state:
    st.session_state.client = OpenAI(api_key=st.secrets["OPEN_API_KEY"])

if "collection" not in st.session_state:
    st.session_state.collection = get_collection()
    ingest_html_if_needed(st.session_state.collection)

SYSTEM = (
    "You are a helpful chatbot answering questions using student organization documents.\n"
    "Use the provided CONTEXT as your primary source.\n"
    "If the answer is not clearly in the context, say you don't see it in the documents and ask a follow-up question.\n"
    "When you do use the context, begin with: 'Based on the student org documents,'\n"
)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM}]

# render chat history
for m in st.session_state.messages:
    if m["role"] == "system":
        continue
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Ask about student orgs...")

if prompt:
    # show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context, sources = retrieve_context(st.session_state.collection, prompt, k=6)

    rag_msg = {
        "role": "system",
        "content": f"CONTEXT:\n{context}\n\nSOURCES: {sources}",
    }

    messages_to_send = [st.session_state.messages[0], rag_msg] + keep_last_5_interactions(st.session_state.messages)[1:]

    completion = st.session_state.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages_to_send,
    )

    answer = completion.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

