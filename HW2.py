import streamlit as st
from openai import OpenAI

st.title("HW 2")
st.write("Upload a document below and ask a question about it – GPT will answer!")

summary_type = st.sidebar.radio("Summary type", ["100 words", "2 connected paragraphs", "5 bullet points"])
use_advanced = st.sidebar.checkbox("Use advanced model")
model = "gpt-4.1" if use_advanced else "gpt-4.1-nano"

openai_api_key = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("OPEN_API_KEY")
if not openai_api_key:
    st.error("Missing API key in Streamlit secrets. Add OPENAI_API_KEY (or OPEN_API_KEY).")
    st.stop()
client = OpenAI(api_key=openai_api_key)

uploaded_file = st.file_uploader("Upload a document (.txt or .md)", type=("txt", "md"))

if uploaded_file:
    document = uploaded_file.read().decode("utf-8", errors="ignore")

    if summary_type == "100 words":
        instruction = "Summarize the document in exactly 100 words."
    elif summary_type == "2 connected paragraphs":
        instruction = "Summarize the document in 2 connected paragraphs."
    else:
        instruction = "Summarize the document in 5 concise bullet points."

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": f"{instruction}\n\nDocument:\n{document}"}],
        stream=True,
    )

    def stream_text(stream):
        for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta

    st.write_stream(stream_text(stream))