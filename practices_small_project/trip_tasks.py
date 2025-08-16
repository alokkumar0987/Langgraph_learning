import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Define the state
class TripState(dict):
    origin: str
    cities: list
    interests: list
    trip_range: str
    result: str

# Initialize LLM
llm = ChatGroq(model="llama3-8b-8192")

# ------------------ Define LangGraph Nodes ------------------

def identify_city_node(state: TripState):
    """Pick best city based on weather, events, costs"""
    query = f"""
    Analyze and select the best city for the trip based 
    on weather, events, and travel costs.
    
    Traveling from: {state['origin']}
    City Options: {state['cities']}
    Trip Date: {state['trip_range']}
    Traveler Interests: {state['interests']}
    """
    resp = llm.invoke(query)
    state["result"] = resp.content
    return state


def gather_city_info_node(state: TripState):
    """Gather in-depth city guide"""
    query = f"""
    As a local expert, compile an in-depth guide for this city.
    Include attractions, hidden gems, cultural hotspots, weather,
    events, and high level costs.
    
    Trip Date: {state['trip_range']}
    Traveling from: {state['origin']}
    Traveler Interests: {state['interests']}
    """
    resp = llm.invoke(query)
    state["result"] = resp.content
    return state


def plan_itinerary_node(state: TripState):
    """Make a full 7-day travel itinerary"""
    query = f"""
    Expand into a full 7-day itinerary with daily schedule, 
    weather forecasts, restaurants, hotels, budget, and packing tips.
    
    Trip Date: {state['trip_range']}
    Traveling from: {state['origin']}
    Traveler Interests: {state['interests']}
    """
    resp = llm.invoke(query)
    state["result"] = resp.content
    return state


# ------------------ Build Graph ------------------

workflow = StateGraph(TripState)

workflow.add_node("identify_city", identify_city_node)
workflow.add_node("gather_city_info", gather_city_info_node)
workflow.add_node("plan_itinerary", plan_itinerary_node)

workflow.set_entry_point("identify_city")
workflow.add_edge("identify_city", "gather_city_info")
workflow.add_edge("gather_city_info", "plan_itinerary")
workflow.add_edge("plan_itinerary", END)

app = workflow.compile()

# ------------------ Run ------------------

