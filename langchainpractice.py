import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.title("🎬 Movie Recommendation Chatbot")

llm = Anthropic(
    model="claude-haiku-4-5-20251001",
    api_key=st.secrets["ANTHROPIC_API_KEY"]
)

prompt = PromptTemplate(
    input_variables=["genre", "mood"],
    template="""You are a helpful movie recommendation assistant.
The user is in the mood for {genre} movies and is feeling {mood}.
Recommend 3 movies and briefly explain why each one fits."""
)

chain = prompt | llm | StrOutputParser()

genre = st.text_input("What genre are you in the mood for?", "action")
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
    template="""You are a helpful movie recommendation assistant. 
You remember everything the user has told you about their preferences.

Conversation so far:
{history}

User: {input}
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

from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

@tool
def get_popular_movies(genre: str) -> str:
    """Get a list of popular movies for a given genre."""
    movies = {
        "action": "Mad Max: Fury Road, John Wick, The Dark Knight",
        "comedy": "The Grand Budapest Hotel, Superbad, Knives Out",
        "horror": "Hereditary, Get Out, A Quiet Place",
        "drama": "The Shawshank Redemption, Parasite, Moonlight",
        "sci-fi": "Interstellar, Arrival, Ex Machina"
    }
    return movies.get(genre.lower(), "Try Inception, The Matrix, or Pulp Fiction")

agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful movie recommendation assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, [get_popular_movies], agent_prompt)
executor = AgentExecutor(agent=agent, tools=[get_popular_movies], verbose=True)

user_input = st.chat_input("Ask the agent for a recommendation...")
if user_input:
    st.chat_message("user").write(user_input)
    response = executor.invoke({"input": user_input})
    st.chat_message("assistant").write(response["output"])


