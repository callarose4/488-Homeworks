
import os
import re
import streamlit as st
import anthropic
import chromadb
import pandas as pd
from chromadb.utils import embedding_functions

st.set_page_config(
    page_title="News Information Bot",
    page_icon="📰",
    layout="wide",
)

CSV_PATH = "news.csv"
DB_PATH = "./chroma_db"
TOP_K = 8

MODELS = {
    "claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5": "claude-sonnet-4-5",
}

SYSTEM_PROMPT = """You are a professional news analyst assistant for a large global law firm.
Your job is to help attorneys and staff monitor news about the firm's clients.

You have access to a curated set of recent news articles. Base your answers ONLY on
the articles provided in each query. Do not fabricate or hallucinate facts.

When asked to rank or find interesting/important news:
- Prioritize legal, regulatory, and financial risk stories
- Consider company significance, novelty, and potential client impact
- Return a numbered ranked list with a brief explanation for each item

When asked to find news about a specific company or topic:
- Return all relevant articles you find in the provided context
- Group by company if multiple companies appear
- Include the article URL so attorneys can read the full piece

Always cite which article(s) you are drawing from using the [Article N] label.
Explain the reasoning behind your choosing of each article for prompted descriptions clearly and concisely. 
Be concise, professional, and accurate. If no relevant articles are found, say so clearly."""

@st.cache_resource
def load_or_build_collection():
    """
    Load the ChromaDB collection from disk if it exists,
    otherwise build it from news.csv. Cached so it only runs once per session.
    """
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=DB_PATH)

    existing = [c.name for c in client.list_collections()]
    if "news" in existing:
        collection = client.get_collection("news", embedding_function=ef)
        return collection


    df = pd.read_csv(CSV_PATH)
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    df = df.dropna(subset=["Document"]).reset_index(drop=True)

    collection = client.create_collection(
        name="news",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [str(i) for i in df.index]
    documents = df["Document"].tolist()
    metadatas = [
        {
            "company": row["company_name"],
            "date": str(row["Date"]),
            "url": str(row["URL"]),
            "days_since_2000": int(row["days_since_2000"]),
        }
        for _, row in df.iterrows()
    ]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return collection


# ── Anthropic client ──────────────────────────────────────────────────────────
@st.cache_resource
def get_anthropic_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        st.error("ANTHROPIC_API_KEY is not set. Please export it before running.")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


# ── RAG helpers ───────────────────────────────────────────────────────────────
def retrieve_articles(query: str, collection, n_results: int = TOP_K) -> list[dict]:
    """Embed the query and return the top-n most similar articles."""
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
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
                "company": meta.get("company", "Unknown"),
                "date": meta.get("date", "")[:10],
                "url": meta.get("url", ""),
                "similarity": round(1 - dist, 4),
            }
        )
    return articles


def format_articles_for_prompt(articles: list[dict]) -> str:
    """Format retrieved articles into a clean context block for the LLM."""
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


# ── Intent detection ──────────────────────────────────────────────────────────
def detect_intent(user_message: str) -> tuple[str, str]:
    """
    Classify the query intent and extract the best search string.
    Returns (intent, search_query).
    Intents: 'rank' | 'topic' | 'general'
    """
    msg_lower = user_message.lower()

    rank_keywords = [
        "interesting", "important", "top news", "most relevant",
        "biggest", "best", "notable", "significant", "latest",
        "rank", "ranked", "highlight", "highlights",
    ]
    if any(kw in msg_lower for kw in rank_keywords):
        return "rank", user_message

    topic_match = re.search(
        r"(?:news about|articles? (?:about|on|regarding)|find|search for|tell me about)\s+(.+)",
        msg_lower,
    )
    if topic_match:
        return "topic", topic_match.group(1).strip()

    return "general", user_message


# ── LLM call ──────────────────────────────────────────────────────────────────
def ask_llm(
    user_message: str,
    context_articles: list[dict],
    model_id: str,
    chat_history: list[dict],
    client: anthropic.Anthropic,
) -> str:
    """Send the user query + retrieved articles to the chosen model."""
    context_text = format_articles_for_prompt(context_articles)

    # Full conversation history plus current turn with injected context
    messages = list(chat_history)
    messages.append(
        {
            "role": "user",
            "content": (
                f"{user_message}\n\n"
                f"=== RETRIEVED ARTICLES ===\n\n"
                f"{context_text}\n\n"
                f"=== END ARTICLES ==="
            ),
        }
    )

    response = client.messages.create(
        model=model_id,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


# ── Streamlit UI ──────────────────────────────────────────────────────────────
def main():
    st.title("📰 News Information Bot")
    st.caption("Ask questions about recent client news. Powered by RAG + Claude.")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        model_label = st.selectbox("Select Model", list(MODELS.keys()))
        model_id = MODELS[model_label]

        st.markdown("---")
        st.markdown("**Example queries:**")
        st.markdown("- *Find the most interesting news*")
        st.markdown("- *Find news about Apple*")
        st.markdown("- *What regulatory issues came up this week?*")
        st.markdown("- *Are there any legal risks for JPMorgan?*")

        st.markdown("---")
        if st.button("🗑️ Clear conversation"):
            st.session_state.messages = []
            st.session_state.raw_history = []
            st.rerun()

    # Load resources (cached)
    with st.spinner("Loading news database..."):
        try:
            collection = load_or_build_collection()
            client = get_anthropic_client()
        except Exception as e:
            st.error(f"Error loading resources: {e}")
            st.stop()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []       # display messages (with article metadata)
    if "raw_history" not in st.session_state:
        st.session_state.raw_history = []    # clean LLM history (no injected context)

    # Render existing chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("articles"):
                with st.expander(f"📄 {len(msg['articles'])} source articles used"):
                    for a in msg["articles"]:
                        st.markdown(
                            f"**{a['company']}** &nbsp;·&nbsp; {a['date']} &nbsp;·&nbsp; "
                            f"similarity: `{a['similarity']}`  \n"
                            f"[{a['url']}]({a['url']})"
                        )

    # Chat input
    if prompt := st.chat_input("Ask about the news..."):

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Retrieve + generate
        with st.chat_message("assistant"):
            with st.spinner("Searching articles and generating response..."):
                intent, search_query = detect_intent(prompt)
                articles = retrieve_articles(search_query, collection, n_results=TOP_K)
                response_text = ask_llm(
                    user_message=prompt,
                    context_articles=articles,
                    model_id=model_id,
                    chat_history=st.session_state.raw_history,
                    client=client,
                )

            st.markdown(response_text)

            with st.expander(f"📄 {len(articles)} source articles used"):
                for a in articles:
                    st.markdown(
                        f"**{a['company']}** &nbsp;·&nbsp; {a['date']} &nbsp;·&nbsp; "
                        f"similarity: `{a['similarity']}`  \n"
                        f"[{a['url']}]({a['url']})"
                    )

        # Save to session state
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response_text,
                "articles": articles,
            }
        )

        # Save lean history for LLM (no context bloat across turns)
        st.session_state.raw_history.append({"role": "user", "content": prompt})
        st.session_state.raw_history.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()