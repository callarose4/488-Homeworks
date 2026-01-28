import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

def read_url_content(url):
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    except requests.RequestException:
        return None

st.title("HW 2")
st.write("Upload a document below and ask a question about it – GPT will answer!")

url = st.text_input(
    "Enter a URL to summarize content from (e.g., a news article or blog post):",
    placeholder="https://example.com/article"
)

summary_type = st.sidebar.radio("Summary type", ["100 words", "2 connected paragraphs", "5 bullet points"])
language = st.sidebar.selectbox(
    "Output language",
    ["English", "French", "Spanish"]
)
use_advanced = st.sidebar.checkbox("Use advanced model")
model = "gpt-4.1" if use_advanced else "gpt-4.1-nano"

openai_api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("OPEN_API_KEY")
if not openai_api_key:
    st.error("Missing API key in Streamlit secrets. Add OPENAI_API_KEY (or OPEN_API_KEY).")
    st.stop()
client = OpenAI(api_key=openai_api_key)


if url:
    page_text = read_url_content(url)

    if summary_type == "100 words":
        instruction = "Summarize the document in exactly 100 words."
    elif summary_type == "2 connected paragraphs":
        instruction = "Summarize the document in 2 connected paragraphs."
    else:
        instruction = "Summarize the document in 5 concise bullet points."
    language_instruction = f"Write the summary in {language}."




    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"{instruction}\n\nDocument:\n{page_text}\n\n{language_instruction}"}],
        stream=True,
    )

    def stream_text(stream):
        for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta

    st.write_stream(stream_text(stream))