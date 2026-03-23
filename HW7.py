import re
import json
import pandas as pd
import streamlit as st
import anthropic
import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime


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
    "You should provide context for 'interesting' news.\n\n"
    "When asked about a specific company or topic, return all relevant articles. "
    "Always include the article URL. "
    "Cite articles using [Article N] labels.\n\n"
    "You have access to a tool called filter_by_date. Use it when the user's query "
    "mentions a specific date, time period, or relative time reference such as "
    "'last week', 'in July', 'before August', or 'recent'. "
    "Call the tool first, then answer based on the filtered articles."
)

TOOLS = [
    {
        "name": "filter_by_date",
        "description": (
            "Filters the news article database to only return articles within a "
            "specified date range. Use this when the user asks about news from a "
            "specific time period, such as 'last week', 'in July', or 'before August 1st'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "The start of the date range in YYYY-MM-DD format. Leave empty if no lower bound.",
                },
                "end_date": {
                    "type": "string",
                    "description": "The end of the date range in YYYY-MM-DD format. Leave empty if no upper bound.",
                },
            },
            "required": [],
        },
    }
]


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
                "date": int(row["Date"].timestamp()) if pd.notna(row["Date"]) else 0,
                "date_str": str(row["Date"])[:10],
                "url": str(row["URL"]),
            }
            for _, row in df.iterrows()
        ],
    )

    return collection


def retrieve_articles(query, collection, date_filter=None):
    where_clause = None

    if date_filter:
        start = date_filter.get("start_date")
        end = date_filter.get("end_date")
        conditions = []
        if start:
            ts_start = int(datetime.strptime(start, "%Y-%m-%d").timestamp())
            conditions.append({"date": {"$gte": ts_start}})
        if end:
            ts_end = int(datetime.strptime(end, "%Y-%m-%d").timestamp()) + 86399
            conditions.append({"date": {"$lte": ts_end}})
        if len(conditions) == 1:
            where_clause = conditions[0]
        elif len(conditions) == 2:
            where_clause = {"$and": conditions}

    query_kwargs = dict(
        query_texts=[query],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )
    if where_clause:
        query_kwargs["where"] = where_clause

    results = collection.query(**query_kwargs)

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
                "date": meta.get("date_str", ""),
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


def stream_reply(client, model, messages, collection, prompt):
    full = ""
    articles = None

    with st.chat_message("assistant"):
        box = st.empty()
        try:
            system = messages[0]["content"]
            non_system = [m for m in messages if m["role"] != "system"]

            # First call — may include a tool use request
            response = client.messages.create(
                model=model,
                system=system,
                messages=non_system,
                tools=TOOLS,
                max_tokens=1500,
            )

            # Handle tool use if the model calls filter_by_date
            if response.stop_reason == "tool_use":
                tool_use_block = next(
                    b for b in response.content if b.type == "tool_use"
                )
                date_filter = tool_use_block.input
                articles = retrieve_articles(prompt, collection, date_filter=date_filter)
                context = format_articles(articles)

                # Feed tool result back and stream the final answer
                tool_result_messages = non_system + [
                    {"role": "assistant", "content": response.content},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_block.id,
                                "content": f"Date filter applied ({date_filter}). Here are the filtered articles:\n\n{context}",
                            }
                        ],
                    },
                ]

                with client.messages.stream(
                    model=model,
                    system=system,
                    messages=tool_result_messages,
                    max_tokens=1500,
                ) as s:
                    for text in s.text_stream:
                        full += text
                        box.markdown(full)

            else:
                # No tool call — standard semantic retrieval, stream normally
                base_articles = retrieve_articles(prompt, collection)
                articles = base_articles
                context = format_articles(base_articles)

                augmented_messages = non_system[:-1] + [
                    {
                        "role": "user",
                        "content": f"{prompt}\n\n=== ARTICLES ===\n\n{context}\n\n=== END ===",
                    }
                ]

                with client.messages.stream(
                    model=model,
                    system=system,
                    messages=augmented_messages,
                    max_tokens=1500,
                ) as s:
                    for text in s.text_stream:
                        full += text
                        box.markdown(full)

        except Exception as e:
            full = f"Error: {e}"
            box.markdown(full)

    return full, articles


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

    # Initial retrieval for the first LLM call (tool may override this)
    initial_articles = retrieve_articles(prompt, collection)
    context = format_articles(initial_articles)

    messages_for_llm = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + st.session_state.history
        + [{"role": "user", "content": f"{prompt}\n\n=== ARTICLES ===\n\n{context}\n\n=== END ==="}]
    )

    reply, articles = stream_reply(
        st.session_state.anthropic_client,
        model_id,
        messages_for_llm,
        collection,
        prompt,
    )

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "articles": articles or initial_articles}
    )
    st.session_state.history.append({"role": "user", "content": prompt})
    st.session_state.history.append({"role": "assistant", "content": reply})