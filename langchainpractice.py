import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

st.title("🎬 Movie Recommendation App")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Your Preferences")

genre = st.sidebar.selectbox(
    "Favorite genre?",
    ["Action", "Comedy", "Horror", "Drama", "Sci-Fi", "Thriller", "Romance"]
)

mood = st.sidebar.selectbox(
    "How are you feeling?",
    ["Excited", "Happy", "Sad", "Bored", "Scared", "Romantic",
     "Curious", "Tense", "Melancholy"]
)

persona = st.sidebar.selectbox(
    "Recommendation style?",
    ["Film Critic", "Casual Friend", "Movie Journalist"]
)

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = init_chat_model(
    "claude-haiku-4-5-20251001",
    temperature=0,
    api_key=st.secrets["ANTHROPIC_API_KEY"]
)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Part B: Basic recommendation chain ───────────────────────────────────────
st.subheader("Get Recommendations")

recommendation_prompt = PromptTemplate(
    input_variables=["genre", "mood", "persona"],
    template="""You are a {persona} giving movie recommendations.
The user likes {genre} movies and is feeling {mood}.
Recommend 3 movies and explain why each one fits.
Match your tone and style to your persona."""
)

recommendation_chain = recommendation_prompt | llm | StrOutputParser()

with st.expander("👀 See what LangChain sends to the model"):
    st.code(recommendation_prompt.format(
        genre=genre,
        mood=mood,
        persona=persona
    ))

if st.button("Get Recommendations"):
    with st.spinner("Finding movies..."):
        response = recommendation_chain.invoke({
            "genre": genre,
            "mood": mood,
            "persona": persona
        })

    st.write(response)

    st.session_state.messages.append(
        HumanMessage(content=f"Recommend {genre} movies for someone feeling {mood}. Style: {persona}")
    )
    st.session_state.messages.append(AIMessage(content=response))

st.divider()

# ── Part C: Follow up chat with memory ───────────────────────────────────────
st.subheader("Ask Follow-Up Questions")

conversation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful movie recommendation assistant.
You have already provided movie recommendations to the user.
Answer any follow up questions they have about the movies,
directors, actors, or anything else related to the recommendations."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

conversation_chain = conversation_prompt | llm | StrOutputParser()

for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    st.chat_message(role).write(msg.content)

user_input = st.chat_input("Ask a follow up question...")
if user_input:
    st.chat_message("user").write(user_input)

    response = conversation_chain.invoke({
        "history": st.session_state.messages,
        "input": user_input
    })

    st.chat_message("assistant").write(response)
    st.session_state.messages.append(HumanMessage(content=user_input))
    st.session_state.messages.append(AIMessage(content=response))

if st.sidebar.button("🗑️ Clear conversation"):
    st.session_state.messages = []
    st.rerun()