import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.title(" Movie Recommendation Chatbot")

llm = ChatAnthropic(
    model="claude-haiku-4-5-20251001",
    api_key=st.secrets["ANTHROPIC_API_KEY"]
)
prompt = PromptTemplate(
    input_variables=["genre", "mood"],
    template="""You are a movie recommendation assistant.
The user likes {genre} movies and is feeling {mood}.
Recommend 3 movies and explain why each one fits."""
)

chain = prompt | llm | StrOutputParser()

genre = st.text_input("Favorite genre?", "action")
mood = st.text_input("How are you feeling?", "excited")

if st.button("Get Recommendations"):
    response = chain.invoke({"genre": genre, "mood": mood})
    st.write(response)
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""You are a movie recommendation assistant.
Remember everything the user tells you about their preferences.

{history}
Human: {input}
Assistant:"""
)

chain = ConversationChain(
    llm=llm,
    memory=st.session_state.memory,
    prompt=prompt
)

for msg in st.session_state.memory.chat_memory.messages:
    role = "user" if msg.type == "human" else "assistant"
    st.chat_message(role).write(msg.content)

user_input = st.chat_input("Tell me what you want to watch...")
if user_input:
    st.chat_message("user").write(user_input)
    response = chain.predict(input=user_input)
    st.chat_message("assistant").write(response)
