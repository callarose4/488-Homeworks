import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent

 
# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    api_key=st.secrets.get("OPENAI_API_KEY")
)
 
# Create a PromptTemplate that takes 'genre' and 'mood' as variables
# and formats them into a specific search query
prompt = PromptTemplate.from_template(
    """Generate a specific web search query to find great movie recommendations.
    Genre: {genre}
    Mood/Atmosphere: {mood}
    Return only the search query string, nothing else.
    Example output: best thriller movies with suspenseful dark atmosphere 2023 2024"""
)
 
# Build the chain: prompt -> llm -> output parser
query_chain = prompt | llm | StrOutputParser()
 
 
# ── PART B: CREATE THE CUSTOM TOOLS ─────────────────────────────────────────
 
# Initialize DuckDuckGo search — no API key needed!
search = DuckDuckGoSearchRun()
 
@tool
def search_movies(query: str) -> str:
    """Search for movie recommendations based on user preferences and genre.
    Use this tool when the user wants to find movies to watch."""
    return search.invoke(query)
 
@tool
def get_movie_reviews(movie_title: str) -> str:
    """Get critic reviews and ratings for a specific movie title.
    Use this tool when the user wants to know what critics think about
    a particular movie, or wants review scores and ratings."""
    return search.invoke(f"{movie_title} review rotten tomatoes critic score")
 
 
# ── PART C: BUILD THE AGENT ──────────────────────────────────────────────────
 
tools = [search_movies, get_movie_reviews]
agent = create_agent(llm, tools=tools)
 
 
# ── STREAMLIT UI ─────────────────────────────────────────────────────────────
 
st.title("🎬 Movie Recommendation Agent")
st.write("Tell me what you're in the mood for and I'll find real recommendations from the web.")
 
# Two input fields for genre and mood
genre = st.text_input("What genre are you in the mood for?", placeholder="e.g. thriller, sci-fi, comedy")
mood = st.text_input("What's the vibe or atmosphere you want?", placeholder="e.g. dark and suspenseful, feel-good, mind-bending")
 
if st.button("Find Movies"):
    if genre and mood:
        with st.spinner("Searching the web..."):
 
            # Step 1: Run the chain to generate a search query
            search_query = query_chain.invoke({"genre": genre, "mood": mood})
 
            # Step 2: Pass the search query to the agent
            response = agent.invoke({
                "messages": [("user", search_query)]
            })
 
            # Display the agent's response
            st.subheader("🍿 Recommendations")
            st.write(response["messages"][-1].content)
 
    else:
        st.warning("Please fill in both genre and mood before searching.")
 
# ── PART D: REVIEW TOOL ──────────────────────────────────────────────────────
 
st.divider()
st.subheader("🎥 Get Reviews for a Specific Movie")
movie_title = st.text_input("Want reviews for a specific movie? Enter the title:")
 
if st.button("Get Reviews"):
    if movie_title:
        with st.spinner("Fetching reviews..."):
            review_response = agent.invoke({
                "messages": [("user", f"Get me critic reviews and ratings for the movie: {movie_title}")]
            })
            st.write(review_response["messages"][-1].content)
    else:
        st.warning("Please enter a movie title.")