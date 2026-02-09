import streamlit as st
from openai import OpenAI
import tiktoken
import requests
from bs4 import BeautifulSoup

# Anthropic is optional at runtime (only needed if you select Anthropic)
try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except Exception:
    ANTHROPIC_AVAILABLE = False


# ----------------------------
# URL reader (re-used from HW2)
# ----------------------------
def read_url_content(url: str) -> str:
    if not url:
        return ""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except Exception:
        return ""


# ----------------------------
# Token helpers (OpenAI-only)
# ----------------------------
def count_tokens(messages, model_name):
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    total = 0
    for m in messages:
        total += len(enc.encode(m.get("content", "")))
        total += 4
    return total


def enforce_max_tokens(messages, model_name, max_tok):
    system = []
    rest = messages[:]

    if rest and rest[0]["role"] == "system":
        system = [rest[0]]
        rest = rest[1:]

    while rest and count_tokens(system + rest, model_name) > max_tok:
        rest.pop(0)

    if system:
        return system + rest if (system + rest) else system
    return rest if rest else [{"role": "user", "content": "Hi"}]


def last_two_user_turns(messages):
    system = [m for m in messages if m["role"] == "system"][:1]

    user_idxs = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if len(user_idxs) <= 2:
        return system + [m for m in messages if m["role"] != "system"]

    start = user_idxs[-2]
    tail = [m for m in messages[start:] if m["role"] != "system"]
    return system + tail


def yes_no_intent(text: str) -> str:
    t = text.strip().lower()
    if t in {"yes", "y", "yeah", "yep", "sure", "yea"}:
        return "yes"
    if t in {"no", "n", "nope", "nah"}:
        return "no"
    return "other"


# ----------------------------
# System prompt builder
# ----------------------------
def build_system_prompt(url_context: str) -> str:
    base = (
        "You are a helpful chatbot. Use the provided URL sources as your primary context.\n"
        "If the sources do not contain enough info, say what is missing.\n\n"
        "Explain answers so a 10-year-old can understand (simple words, short sentences).\n"
        "After answering, ask exactly: Do you want more info?\n"
        "If the user says Yes, give more info about the last question and ask again.\n"
        "If the user says No, respond exactly: Okay—what can I help you with?\n"
    )

    if url_context.strip():
        return base + "\n\nSOURCES:\n" + url_context
    return base + "\n\nSOURCES:\n(No URLs loaded.)"


# ----------------------------
# Streaming reply (OpenAI + Anthropic)
# ----------------------------
def stream_assistant_reply(provider, openai_client, anthropic_client, model_to_use, messages_to_send):
    full_response = ""

    with st.chat_message("assistant"):
        box = st.empty()

        if provider == "OpenAI":
            stream = openai_client.chat.completions.create(
                model=model_to_use,
                messages=messages_to_send,
                stream=True,
            )
            for event in stream:
                delta = event.choices[0].delta.content
                if delta:
                    full_response += delta
                    box.markdown(full_response)

        else:
            if anthropic_client is None:
                msg = "Anthropic client not configured. Add ANTHROPIC_API_KEY to Streamlit secrets."
                box.markdown(msg)
                return msg

            # Anthropic uses separate `system=` plus messages without system role
            system_text = ""
            if messages_to_send and messages_to_send[0]["role"] == "system":
                system_text = messages_to_send[0]["content"]

            non_system = [m for m in messages_to_send if m["role"] != "system"]

            stream = anthropic_client.messages.stream(
                model=model_to_use,
                system=system_text,
                messages=[{"role": m["role"], "content": m["content"]} for m in non_system],
                max_tokens=800,
            )

            with stream as s:
                for text in s.text_stream:
                    full_response += text
                    box.markdown(full_response)

    return full_response


# ============================
# UI
# ============================
st.title("HW 3 — Streaming URL Chatbot")

st.write(
    """
This chatbot streams replies and remembers the conversation using a token-limited buffer.
It always keeps the URL text inside a system prompt, so the sources are never discarded.
"""
)

# Sidebar: URLs
st.sidebar.header("Sources (up to 2 URLs)")
url1 = st.sidebar.text_input("URL 1", placeholder="https://example.com/page")
url2 = st.sidebar.text_input("URL 2 (optional)", placeholder="https://example.com/page2")
load_urls = st.sidebar.button("Load URLs")

# Sidebar: vendor + model
st.sidebar.header("Model")
provider = st.sidebar.selectbox("Vendor", ["OpenAI", "Anthropic"])

if provider == "OpenAI":
    model_to_use = st.sidebar.selectbox("OpenAI model", ["gpt-4.1", "gpt-4o"])
else:
    # Pick premium Anthropic models (these names must match your installed SDK/version access)
    model_to_use = st.sidebar.selectbox(
        "Anthropic model",
        ["claude-3-opus-20240229", "claude-3-5-sonnet-20241022"],
        disabled=not ANTHROPIC_AVAILABLE,
        help="Install anthropic and add ANTHROPIC_API_KEY in secrets to use this."
    )

# keep max_tokens defined in the app (memory requirement)
MAX_TOKENS = 1200

# Session state: clients
if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=st.secrets["OPEN_API_KEY"])

if "anthropic_client" not in st.session_state:
    if ANTHROPIC_AVAILABLE and "ANTHROPIC_API_KEY" in st.secrets:
        st.session_state.anthropic_client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    else:
        st.session_state.anthropic_client = None

# Session state: URL context
if "url_context" not in st.session_state:
    st.session_state.url_context = ""
if "loaded_urls" not in st.session_state:
    st.session_state.loaded_urls = []

if load_urls:
    texts = []
    loaded = []

    if url1.strip():
        t1 = read_url_content(url1.strip())
        if t1:
            texts.append(f"URL 1 ({url1.strip()}):\n{t1}")
            loaded.append(url1.strip())
        else:
            st.sidebar.warning("Could not read URL 1.")

    if url2.strip():
        t2 = read_url_content(url2.strip())
        if t2:
            texts.append(f"URL 2 ({url2.strip()}):\n{t2}")
            loaded.append(url2.strip())
        else:
            st.sidebar.warning("Could not read URL 2.")

    st.session_state.url_context = "\n\n".join(texts) if texts else ""
    st.session_state.loaded_urls = loaded
    st.sidebar.success(f"Loaded {len(loaded)} URL(s).")

if st.session_state.loaded_urls:
    st.sidebar.caption("Loaded URLs:")
    for u in st.session_state.loaded_urls:
        st.sidebar.write(f"- {u}")

# State for Part C behavior
if "awaiting_more_info" not in st.session_state:
    st.session_state.awaiting_more_info = False
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

# Messages with SYSTEM that is never discarded
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": build_system_prompt(st.session_state.url_context)}]
else:
    # Always keep system prompt up to date (still never discarded)
    if st.session_state.messages and st.session_state.messages[0]["role"] == "system":
        st.session_state.messages[0]["content"] = build_system_prompt(st.session_state.url_context)
    else:
        st.session_state.messages.insert(0, {"role": "system", "content": build_system_prompt(st.session_state.url_context)})

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "system":
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask me anything about the URL(s)!")

if prompt:
    # show/store user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    intent = yes_no_intent(prompt)

    # Decide what we want the model to do
    if st.session_state.awaiting_more_info and intent == "no":
        bot_text = "Okay—what can I help you with?"
        with st.chat_message("assistant"):
            st.markdown(bot_text)
        st.session_state.messages.append({"role": "assistant", "content": bot_text})
        st.session_state.awaiting_more_info = False
        st.session_state.last_question = ""

    else:
        # If awaiting and user said yes, add followup tied to last_question
        if st.session_state.awaiting_more_info and intent == "yes":
            st.session_state.messages.append({
                "role": "user",
                "content": f"Give more info about: {st.session_state.last_question}"
            })
            # keep awaiting_more_info = True

        # Otherwise treat as a new question
        elif not st.session_state.awaiting_more_info or intent == "other":
            st.session_state.last_question = prompt
            st.session_state.awaiting_more_info = True

        # Build messages to send (buffer + system protected)
        messages_to_send = last_two_user_turns(st.session_state.messages)

        # Token buffer only makes sense for OpenAI here (tiktoken)
        if provider == "OpenAI":
            messages_to_send = enforce_max_tokens(messages_to_send, model_to_use, MAX_TOKENS)
            tokens_sent = count_tokens(messages_to_send, model_to_use)
            st.sidebar.write(f"Tokens sent this request: {tokens_sent}")
        else:
            st.sidebar.write("Tokens sent this request: (tiktoken not used for Anthropic)")

        # Streaming call
        full_response = stream_assistant_reply(
            provider,
            st.session_state.openai_client,
            st.session_state.anthropic_client,
            model_to_use,
            messages_to_send
        )
        st.session_state.messages.append({"role": "assistant", "content": full_response})
