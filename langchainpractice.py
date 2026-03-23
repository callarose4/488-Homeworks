import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.title(" Movie Recommendation App")

# Initialize LLM
llm = init_chat_model(
    "claude-haiku-4-5-20251001",
    temperature=0
)

# Sidebar inputs
st.sidebar.header("Your Preferences")

genre = st.sidebar.selectbox(
    "Favorite genre?",
    ["Action", "Comedy", "Horror", "Drama", "Sci-Fi", "Thriller", "Romance"]
)

mood = st.sidebar.selectbox(
    "How are you feeling?",
    ["Excited", "Happy", "Sad", "Bored", "Scared", "Romantic", "Curious", "Tense", "Melancholy"]
)

persona = st.sidebar.selectbox(
    "Recommendation style?",
    ["Film Critic", "Casual Friend", "Movie Journalist"]
)

# Prompt template
prompt = PromptTemplate(
    input_variables=["genre", "mood", "persona"],
    template="""You are a {persona} giving movie recommendations.
The user likes {genre} movies and is feeling {mood}.
Recommend 3 movies and explain why each one fits.
Match your tone and style to your persona."""
)

# Build chain
chain = prompt | llm | StrOutputParser()

# Button and response
if st.button("Get Recommendations"):
    with st.spinner("Finding movies..."):
        response = chain.invoke({
            "genre": genre,
            "mood": mood,
            "persona": persona
        })
    st.write(response)