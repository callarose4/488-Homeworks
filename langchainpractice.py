import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.title("🎬 Movie Recommendation Chatbot")

llm = init_chat_model(
    "claude-haiku-4-5-20251001",
    temperature=0
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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

if "messages" not in st.session_state:
    st.session_state.messages = []

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful movie recommendation assistant. 
Remember everything the user tells you about their preferences."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

user_input = st.chat_input("Tell me what you want to watch...")
if user_input:
    st.chat_message("user").write(user_input)

    response = chain.invoke({
        "history": st.session_state.messages,
        "input": user_input
    })

    st.chat_message("assistant").write(response)
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.messages.append(AIMessage(content=response))