from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
import streamlit as st

# Define system prompt
SYSTEM_PROMPT = """You are a movie recommendation expert who knows everything about film.

You have access to two tools:
- get_mood_from_vibe: use this when a user describes a feeling or vibe rather than a specific genre
- get_movies_by_genre: use this to get movie recommendations for a specific genre

If a user asks for a movie recommendation, make sure you know what genre fits them.
If they describe a feeling or vibe, use get_mood_from_vibe first to figure out the right genre, then use get_movies_by_genre."""

# Define context schema
@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

# Define tools
@tool
def get_mood_from_vibe(vibe: str) -> str:
    """Translate a user's mood or vibe into a movie genre."""
    vibes = {
        "happy": "comedy",
        "sad": "drama",
        "scared": "horror",
        "bored": "action",
        "romantic": "romance",
        "curious": "sci-fi",
        "tense": "thriller",
        "excited": "action",
        "melancholy": "drama",
        "adventurous": "action"
    }
    vibe_lower = vibe.lower()
    for key in vibes:
        if key in vibe_lower:
            return vibes[key]
    return "drama"

@tool
def get_movies_by_genre(genre: str) -> str:
    """Get movie recommendations for a specific genre."""
    movies = {
        "action": "Mad Max: Fury Road, John Wick, The Dark Knight, Mission Impossible",
        "comedy": "The Grand Budapest Hotel, Superbad, Knives Out, Game Night",
        "horror": "Hereditary, Get Out, A Quiet Place, Midsommar",
        "drama": "The Shawshank Redemption, Parasite, Moonlight, Marriage Story",
        "sci-fi": "Interstellar, Arrival, Ex Machina, Dune",
        "thriller": "Gone Girl, Prisoners, Zodiac, No Country for Old Men",
        "romance": "Before Sunrise, Eternal Sunshine, La La Land, Normal People"
    }
    return movies.get(genre.lower(), "Try Inception, The Matrix, or Pulp Fiction")

# Configure model
model = init_chat_model(
    "claude-haiku-4-5-20251001",
    temperature=0
)

# Define response format
@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # The movie recommendation (always required)
    recommendation: str
    # The genre detected, if available
    genre: str | None = None

# Set up memory
checkpointer = InMemorySaver()

# Create agent
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_mood_from_vibe, get_movies_by_genre],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

# Streamlit UI
st.title("🎬 Movie Recommendation Agent")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "1"

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle input
user_input = st.chat_input("What are you in the mood for?")
if user_input:
    st.chat_message("user").write(user_input)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        context=Context(user_id="1")
    )

    structured = response["structured_response"]
    answer = structured.recommendation
    if structured.genre:
        answer = f"**Genre detected: {structured.genre}**\n\n{answer}"

    st.chat_message("assistant").write(answer)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": answer})