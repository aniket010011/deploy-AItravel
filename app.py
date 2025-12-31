import streamlit as st
import json
import requests
import os

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="Agentic AI Travel Planner",
    layout="wide"
)

st.title("✈️ Agentic AI Travel Planning Assistant")
st.markdown(
    "Generate a **day-wise travel itinerary** with "
    "**flight, hotel, places, weather, budget**, and "
    "**explanations for each choice**."
)

# =========================================================
# Load OpenAI API Key (Streamlit Secrets)
# =========================================================
if "OPENAI_API_KEY" not in st.secrets:
    st.error("❌ OPENAI_API_KEY not found in Streamlit Secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# =========================================================
# Load Data
# =========================================================
@st.cache_data
def load_data():
    with open("flights.json") as f:
        flights = json.load(f)
    with open("hotels.json") as f:
        hotels = json.load(f)
    with open("places.json") as f:
        places = json.load(f)
    return flights, hotels, places

flights, hotels, places = load_data()

# =========================================================
# LLM Setup
# =========================================================
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0
)

# =========================================================
# Prompt
# =========================================================
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an AGENTIC travel planning assistant.\n\n"
            "Generate a DAY-WISE itinerary (Day 1, Day 2, Day 3) including:\n"
            "- Selected flight\n"
            "- Selected hotel\n"
            "- Places to visit each day\n"
            "- Weather summary\n"
            "- Budget breakdown\n\n"
            "IMPORTANT:\n"
            "After selecting the flight and hotel, explain clearly:\n"
            "• Why this flight was chosen\n"
            "• Why this hotel was chosen\n\n"
            "Base explanations on price, rating, convenience, and suitability "
            "for a budget trip."
        ),
        ("human", "{input}")
    ]
)

agent = (
    {"input": RunnablePassthrough()}
    | prompt
    | llm
)

# =========================================================
# UI Inputs
# =========================================================
col1, col2, col3 = st.columns(3)

with col1:
    source_city = st.text_input("Source City", "Delhi")

with col2:
    destination_city = st.text_input("Destination City", "Goa")

with col3:
    days = st.number_input("Number of Days", 3, 7, 3)

# =========================================================
# Generate Plan
# =========================================================
if st.button("🧳 Generate Travel Plan"):
    query = (
        f"Plan a {days} day budget trip from {source_city} to {destination_city} "
        "with flights, hotels, places, weather and budget."
    )

    with st.spinner("Planning your trip..."):
        try:
            response = agent.invoke(query)
            st.markdown(response.content)
        except Exception as e:
            st.error(f"❌ Error generating plan: {e}")

# =========================================================
# Footer
# =========================================================
st.markdown("---")
st.caption(
    "Built using LangChain + OpenAI | "
    "Agentic AI Travel Planner"
)
