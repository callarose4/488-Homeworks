# --- SQLite fix for Chroma (MUST be first) ---
import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import re
from pathlib import Path

import chromadb
import streamlit as st
from bs4 import BeautifulSoup
from chromadb.utils import embedding_functions
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
HTML_DIR = BASE_DIR / "HW4-HTML"
DB_DIR = BASE_DIR / "chroma_hw4_db"


def html_to_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


# Chunking method: EXACTLY 2 chunks per document.
# Why: HW requires two mini-documents per HTML; splitting at the midpoint is deterministic and keeps
# both the beginning and end of each document searchable in retrieval.
def two_chunks(text: str) -> list[str]:
    mid = max(1, len(text) // 2)
    return [text[:mid].strip(), text[mid:].strip()]


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=st.secrets["OPEN_API_KEY"],
        model_name="text-embedding-3-small",
    )
    return client.get_or_create_collection("hw4_html", embedding_function=ef)


def ingest_if_needed(col):
    # Only build once (safe for reruns)
    if col.count() > 0:
        return

    files = sorted(HTML_DIR.glob("*.html"))
    if not files:
        st.error(f"No HTML files found in: {HTML_DIR}")
        st.stop()

    ids, docs, metas = [], [], []
    for f in files:
        text = html_to_text(f)

        # Skip totally empty docs (prevents junk chunks)
        if not text:
            continue

        c1, c2 = two_chunks(text)
        ids.extend([f"{f.name}::1", f"{f.name}::2"])
        docs.extend([c1, c2])
        metas.extend([{"source": f.name, "part": 1}, {"source": f.name, "part": 2}])

    col.add(ids=ids, documents=docs, metadatas=metas)


def retrieve(col, query: str, k: int = 6):
    res = col.query(query_texts=[query], n_results=k, include=["documents", "metadatas"])
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    sources = sorted({m.get("source", "") for m in metas if m})
    context = "\n\n---\n\n".join(docs)
    return context, sources


def last_5_interactions(messages):
    # keep system + last 10 messages (5 user + 5 assistant)
    system = messages[:1]
    rest = messages[1:]
    return system + rest[-10:]


# ------------------ UI ------------------
st.title("HW4: Student Orgs Chatbot (RAG)")

col = get_collection()
ingest_if_needed(col)

client = OpenAI(api_key=st.secrets["OPEN_API_KEY"])

SYSTEM = (
    "You answer questions about student organizations.\n"
    "Use the provided CONTEXT as evidence.\n"
    "If the context does not contain the answer, say what you *can* infer from it and ask a follow-up.\n"
    "Start answers with: 'Based on the student org documents,'\n"
)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": SYSTEM}]

for m in st.session_state.messages[1:]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

q = st.chat_input("Ask a question...")

if q:
    st.session_state.messages.append({"role": "user", "content": q})
    with st.chat_message("user"):
        st.markdown(q)

    context, sources = retrieve(col, q, k=6)

    rag = {"role": "system", "content": f"CONTEXT:\n{context}\n\nSOURCES: {sources}"}
    msgs = [st.session_state.messages[0], rag] + last_5_interactions(st.session_state.messages)[1:]

    resp = client.chat.completions.create(model="gpt-4o-mini", messages=msgs)
    ans = resp.choices[0].message.content

    st.session_state.messages.append({"role": "assistant", "content": ans})
    with st.chat_message("assistant"):
        st.markdown(ans)
