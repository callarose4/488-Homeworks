import streamlit as st
from openai import OpenAI
import requests
import time
from bs4 import BeautifulSoup
from anthropic import Anthropic

def read_url_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}  # NEW (minimal)
        response = requests.get(url, headers=headers, timeout=15)  # CHANGED
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except requests.RequestException:
        return None

st.title("HW 2")
st.write("Upload a web page below and ask a question about it – GPT will answer!")

url = st.text_input(
    "Enter a URL to summarize content from (e.g., a news article or blog post):",
    placeholder="https://example.com/article"
)

# NEW (minimal): choose provider
provider = st.sidebar.selectbox("LLM provider", ["OpenAI", "Claude"])

summary_type = st.sidebar.radio("Summary type", ["100 words", "2 connected paragraphs", "5 bullet points"])
language = st.sidebar.selectbox("Output language", ["English", "French", "Spanish"])
use_advanced = st.sidebar.checkbox("Use advanced model")

# Models (minimal)
openai_model = "gpt-4.1" if use_advanced else "gpt-4.1-nano"
claude_model = "claude-3-opus-20240229" if use_advanced else "claude-3-haiku-20240307"

# Keys (minimal)
openai_api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("OPEN_API_KEY")
anthropic_api_key = st.secrets.get("ANTHROPIC_API_KEY")

if url:
    page_text = read_url_content(url)

    if not page_text:
        st.error("Could not extract readable text from that URL. Try a different page.")
        st.stop()

    if summary_type == "100 words":
        instruction = "Summarize the webpage in exactly 100 words."
    elif summary_type == "2 connected paragraphs":
        instruction = "Summarize the webpage in exactly 2 connected paragraphs."
    else:
        instruction = "Summarize the webpage in exactly 5 concise bullet points."

    language_instruction = f"Write the summary in {language}."
    prompt = f"{instruction}\n{language_instruction}\n\nWebpage text:\n{page_text}"

    if provider == "OpenAI":
        if not openai_api_key:
            st.error("Missing OpenAI key in Streamlit secrets. Add OPENAI_API_KEY (or OPEN_API_KEY).")
            st.stop()

        client = OpenAI(api_key=openai_api_key)

        stream = client.chat.completions.create(
            model=openai_model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=1  # FIXED typo
        )

        response = st.write_stream(stream)  # keep this
        st.write("This was the response:")
        st.write(response)

    else:  # Claude
        if not anthropic_api_key:
            st.error("Missing Anthropic key in Streamlit secrets. Add ANTHROPIC_API_KEY.")
            st.stop()

        client = Anthropic(api_key=anthropic_api_key)

        # Claude streaming (with 1 retry if overloaded)
        for attempt in range(2):
            try:
                with client.messages.stream(
                    model=claude_model,
                    max_tokens=700,
                    messages=[{"role": "user", "content": prompt}],
                ) as stream:
                    response = st.write_stream(stream.text_stream)
                break  # success
            except Exception as e:
                if "overloaded" in str(e).lower() and attempt == 0:
                    st.warning("Claude is overloaded right now. Retrying once...")
                    time.sleep(2)
                else:
                    st.error(f"Claude request failed: {e}")
                    st.stop()