import streamlit as st
from openai import OpenAI
import tiktoken
import requests
from bs4 import BeautifulSoup


try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except Exception:
    ANTHROPIC_AVAILABLE = False


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


def count_tokens(messages, model_name):
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    total = 0
    for m in messages:
        total += len(enc.encode(m.get("content", ""))) + 4
    return total


def enforce_max_tokens(messages, model_name, max_tok):
    system = []
    rest = messages[:]

    if rest and rest[0]["role"] == "system":
        system = [rest[0]]
        rest = rest[1:]

    while rest and count_tokens(system + rest, model_name) > max_tok:
        rest.pop(0)

    return system + rest if system else rest


def last_two_user_turns(messages):
    system = [m for m in messages if m["role"] == "system"][:1]
    user_idxs = [i for i, m in enumerate(messages) if m["role"] == "user"]

    if len(user_idxs) <= 2:
        return system + [m for m in messages if m["role"] != "system"]

    start = user_idxs[-2]
    return system + [m for m in messages[start:] if m["role"] != "system"]


def yes_no_intent(text: str) -> str:
    t = text.strip().lower()
    if t in {"yes", "y", "yeah", "yep", "sure"}:
        return "yes"
    if t in {"no", "n", "nope", "nah"}:
        return "no"
    return "other"



def build_system_prompt(url_context: str) -> str:
    base = (
        "You are a helpful chatbot. Use the provided URL sources as your primary context.\n"
        "If the sources do not contain enough info, say what is missing.\n\n"
        "Explain answers so a 10-year-old can understand.\n"
        "After answering, ask exactly: Do you want more info?\n"
        "If the user says Yes, give more info and ask again.\n"
        "If the user says No, respond exactly: Okay—what can I help you with?\n"
    )
    return base + "\n\nSOURCES:\n" + (url_context if url_context else "(No URLs loaded.)")

def stream_assistant_reply(provider, openai_client, anthropic_client, model, messages):
    full = ""

    with st.chat_message("assistant"):
        box = st.empty()

        
        if provider == "OpenAI":
            stream = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
            )
            for event in stream:
                delta = event.choices[0].delta.content
                if delta:
                    full += delta
                    box.markdown(full)
            return full

        
        if anthropic_client is None:
            msg = "Anthropic not configured. Add ANTHROPIC_API_KEY."
            box.markdown(msg)
            return msg

        system_text = messages[0]["content"]
        non_system = [m for m in messages if m["role"] != "system"]

        try:
            stream = anthropic_client.messages.stream(
                model=model,
                system=system_text,
                messages=[{"role": m["role"], "content": m["content"]} for m in non_system],
                max_tokens=800,
            )
            with stream as s:
                for text in s.text_stream:
                    full += text
                    box.markdown(full)
            return full
        except Exception as e:
            msg = f"Anthropic error: {e}"
            box.markdown(msg)
            return msg



st.title("HW 3 — Streaming URL Chatbot")

st.write(
    "This chatbot streams replies, remembers conversation with a token buffer, "
    "and always keeps URL content in system memory."
)


st.sidebar.header("Sources")
url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2 (optional)")
load_urls = st.sidebar.button("Load URLs")


st.sidebar.header("Model")
provider = st.sidebar.selectbox("Vendor", ["OpenAI", "Anthropic"])

if provider == "OpenAI":
    model_to_use = st.sidebar.selectbox("Model", ["gpt-4.1", "gpt-4o"])
else:
    model_to_use = st.sidebar.selectbox(
        "Model",
        [
            "claude-opus-4-20250514", 
            "claude-sonnet-4-20250514"
        ],
        disabled=not ANTHROPIC_AVAILABLE,
    )

MAX_TOKENS = 1200


st.session_state.setdefault("openai_client", OpenAI(api_key=st.secrets["OPEN_API_KEY"]))
st.session_state.setdefault(
    "anthropic_client",
    Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
    if ANTHROPIC_AVAILABLE and "ANTHROPIC_API_KEY" in st.secrets
    else None,
)


st.session_state.setdefault("url_context", "")
st.session_state.setdefault("loaded_urls", [])

if load_urls:
    def load_one(url: str, label: str):
        url = url.strip()
        if not url:
            return None

        text = read_url_content(url)

        # Debug
        st.sidebar.write(f"{label} fetched chars: {len(text)}")

        if not text:
            st.sidebar.warning(f"Could not read {label}.")
            return None

        # Keep a short label + content
        return {
            "url": url,
            "label": label,
            "text": text
        }

    data = []
    for label, url in [("URL 1", url1), ("URL 2", url2)]:
        item = load_one(url, label)
        if item:
            data.append(item)

    st.session_state.loaded_urls = [d["url"] for d in data]

    # Build ONE combined context (both URLs)
    st.session_state.url_context = "\n\n".join(
        [f'{d["label"]} ({d["url"]}):\n{d["text"]}' for d in data]
    )

    st.sidebar.success(f"Loaded {len(st.session_state.loaded_urls)} URL(s).")

# Chat state
st.session_state.setdefault("awaiting_more_info", False)
st.session_state.setdefault("last_question", "")

# Messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": build_system_prompt(st.session_state.url_context)}
    ]
else:
    st.session_state.messages[0]["content"] = build_system_prompt(st.session_state.url_context)

# Render history
for msg in st.session_state.messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

prompt = st.chat_input("Ask me something!")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    intent = yes_no_intent(prompt)

    if st.session_state.awaiting_more_info and intent == "no":
        reply = "Okay—what can I help you with?"
        st.session_state.awaiting_more_info = False
    else:
        if intent == "yes":
            st.session_state.messages.append(
                {"role": "user", "content": f"Give more info about: {st.session_state.last_question}"}
            )
        else:
            st.session_state.last_question = prompt
            st.session_state.awaiting_more_info = True

        msgs = last_two_user_turns(st.session_state.messages)
        if provider == "OpenAI":
            msgs = enforce_max_tokens(msgs, model_to_use, MAX_TOKENS)
            st.sidebar.write(f"Tokens: {count_tokens(msgs, model_to_use)}")

        reply = stream_assistant_reply(
            provider,
            st.session_state.openai_client,
            st.session_state.anthropic_client,
            model_to_use,
            msgs,
        )

    st.session_state.messages.append({"role": "assistant", "content": reply})
