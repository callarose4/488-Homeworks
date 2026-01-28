import streamlit as st
import fitz
from openai import OpenAI

def read_pdf(uploaded_file):
    text = ""
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in pdf: 
        text += page.get_text()
    return text

# Show title and description.
st.title("MY HW Document question answering")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    st.stop()

# Validating the key immediately
try:
    client = OpenAI(api_key=openai_api_key)
    client.models.list() #simple validation call
    st.success("API key validated.")

    

except Exception:
    st.error("Invalid API key. Please check and try again.")
    st.stop()


# File uploader and question input
uploaded_file = st.file_uploader("Upload your document", type=["pdf", "txt"])
question = st.text_input("Ask a question about your document:")

if uploaded_file and question:
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == "txt":
        document = uploaded_file.read().decode("utf-8")
    elif file_extension == "pdf":
        document = read_pdf(uploaded_file)
    else: 
        st.error("Unsupported file type.")
        st.stop()

    messages = [
        {
            "role": "user",
            "content": f"Here's a document: {document} \n\n---\n\n {question}",
        } 
    ]

    stream = client.chat.completions.create(
        model="gpt-5-nano",
        messages=messages,
        stream=True,
    )

    # Stream the response to the app using `st.write_stream`.
    st.write_stream(stream)