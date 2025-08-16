


from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import TypedDict, List

from tools.search_tools import SearchTools

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
llm = ChatGroq(model="llama3-8b-8192")

# -------- Define Agents as callable functions -------- #
class TripAgents:

    def city_selection_agent(self):
        def node(state):
            query = f"""
            Select the best city based on weather, season, and prices.
            Traveler preferences: {state.interests if hasattr(state, 'interests') else 'N/A'}
            City options: {state.cities if hasattr(state, 'cities') else 'N/A'}
            """
            resp = llm.invoke(query)
            state.result = resp.content
            return state
        return node

    def local_expert(self):
        def node(state):
            query = f"""
            Provide the best insights about the selected city: {state.result if hasattr(state, 'result') else 'N/A'}
            Include attractions, culture, hidden gems, and costs.
            """
            resp = llm.invoke(query)
            state.result = resp.content
            return state
        return node

    def travel_concierge(self):
        def node(state):
            query = f"""
            Create a detailed travel itinerary for the selected city: {state.result if hasattr(state, 'result') else 'N/A'}
            Include daily schedule, restaurants, hotels, budget, and packing tips.
            """
            resp = llm.invoke(query)
            state.result = resp.content
            return state
        return node


# -------- LangGraph Workflow -------- #
def build_trip_graph():
    trip_agents = TripAgents()

    workflow = StateGraph()

    # Add nodes = agent functions
    workflow.add_node("city_selection", trip_agents.city_selection_agent())
    workflow.add_node("local_expert", trip_agents.local_expert())
    workflow.add_node("travel_concierge", trip_agents.travel_concierge())

    # Define execution order
    workflow.add_edge("city_selection", "local_expert")
    workflow.add_edge("local_expert", "travel_concierge")
    workflow.add_edge("travel_concierge", END)

    # Starting point
    workflow.set_entry_point("city_selection")

    return workflow.compile()
