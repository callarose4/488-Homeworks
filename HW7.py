import re
import pandas as pd
import streamlit as st
import anthropic
import chromadb
from chromadb.utils import embedding_functions


CSV_PATH = "news.csv"
DB_PATH = "./chroma_db"
TOP_K = 8

MODELS = {
    "claude-haiku-4-5-20251001 (low-cost)": "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5 (high-cost)": "claude-sonnet-4-5",
}

SYSTEM_PROMPT = (
    "You are a news reporting bot for a large global law firm. "
    "You help attorneys monitor news about the firm's clients.\n\n"
    "You will be given a set of news articles as context. "
    "Answer based only on those articles — do not make up facts.\n\n"
    "When asked for interesting or important news, return a ranked numbered list. "
    "For each item explain why it is significant.\n\n"
    "You should provide context for “interesting” news.\n\n"
    "When asked about a specific company or topic, return all relevant articles. "
    "Always include the article URL. "
    "Cite articles using [Article N] labels."
)


@st.cache_resource
def load_or_build_collection():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=DB_PATH)
    existing = [c.name for c in client.list_collections()]

    if "news" in existing:
        return client.get_collection("news", embedding_function=ef)

    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = df.dropna(subset=["Document"]).reset_index(drop=True)

    collection = client.create_collection(
        name="news",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[str(i) for i in df.index],
        documents=df["Document"].tolist(),
        metadatas=[
            {
                "company": row["company_name"],
                "date": str(row["Date"]),
                "url": str(row["URL"]),
            }
            for _, row in df.iterrows()
        ],
    )

    return collection


def retrieve_articles(query, collection):
    results = collection.query(
        query_texts=[query],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    articles = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        articles.append(
            {
                "text": doc,
                "company": meta.get("company", ""),
                "date": meta.get("date", "")[:10],
                "url": meta.get("url", ""),
                "similarity": round(1 - dist, 4),
            }
        )
    return articles


def format_articles(articles):
    blocks = []
    for i, a in enumerate(articles, 1):
        blocks.append(
            f"[Article {i}]\n"
            f"Company: {a['company']}\n"
            f"Date: {a['date']}\n"
            f"URL: {a['url']}\n"
            f"Content: {a['text']}"
        )
    return "\n\n---\n\n".join(blocks)


def stream_reply(client, model, messages):
    full = ""
    with st.chat_message("assistant"):
        box = st.empty()
        try:
            system = messages[0]["content"]
            non_system = [m for m in messages if m["role"] != "system"]
            with client.messages.stream(
                model=model,
                system=system,
                messages=non_system,
                max_tokens=1500,
            ) as s:
                for text in s.text_stream:
                    full += text
                    box.markdown(full)
        except Exception as e:
            full = f"Error: {e}"
            box.markdown(full)
    return full


# ── App ───────────────────────────────────────────────────────────────────────

st.title("News Information Bot")

st.sidebar.header("Model")
model_label = st.sidebar.selectbox("Select Model", list(MODELS.keys()))
model_id = MODELS[model_label]

if st.sidebar.button("🗑️ Clear conversation"):
    st.session_state.messages = []
    st.session_state.history = []
    st.rerun()

with st.spinner("Loading news database..."):
    try:
        collection = load_or_build_collection()
    except Exception as e:
        st.error(f"Failed to load database: {e}")
        st.stop()

st.session_state.setdefault(
    "anthropic_client",
    anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"]),
)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("history", [])

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("articles"):
            with st.expander(f"📄 {len(msg['articles'])} articles retrieved"):
                for a in msg["articles"]:
                    st.markdown(
                        f"**{a['company']}** · {a['date']} · similarity: `{a['similarity']}`  \n"
                        f"[{a['url']}]({a['url']})"
                    )

prompt = st.chat_input("Ask about the news...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    articles = retrieve_articles(prompt, collection)
    context = format_articles(articles)

    messages_for_llm = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + st.session_state.history
        + [{"role": "user", "content": f"{prompt}\n\n=== ARTICLES ===\n\n{context}\n\n=== END ==="}]
    )

    reply = stream_reply(
        st.session_state.anthropic_client,
        model_id,
        messages_for_llm,
    )

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "articles": articles}
    )
    st.session_state.history.append({"role": "user", "content": prompt})
    st.session_state.history.append({"role": "assistant", "content": reply})