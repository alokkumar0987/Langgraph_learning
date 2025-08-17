import json
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_community.chat_models import ChatOllama
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv
import os
from agents import (
    generate_itinerary,
    recommend_activities,
    fetch_useful_links,
    weather_forecaster,
    packing_list_generator,
    food_culture_recommender,
    chat_agent
)

# Load environment variables
load_dotenv()

class GraphState(TypedDict):
    preferences_text: str
    preferences: dict
    itinerary: str
    activity_suggestions: str
    useful_links: list[dict]
    weather_forecast: str
    packing_list: str
    food_culture_info: str
    chat_history: Annotated[list[dict], "List of question-response pairs"]
    user_question: str
    chat_response: str

def initialize_agents():
    """Initialize all required components for the travel agent system."""
    try:
        llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")
        search = TavilySearchResults()
        return llm, search
    except Exception as e:
        raise RuntimeError(f"Initialization failed: {str(e)}")

def create_workflow():
    """Create and configure the LangGraph workflow."""
    workflow = StateGraph(GraphState)
    
    # Add all nodes
    workflow.add_node("generate_itinerary", generate_itinerary.generate_itinerary)
    workflow.add_node("recommend_activities", recommend_activities.recommend_activities)
    workflow.add_node("fetch_useful_links", fetch_useful_links.fetch_useful_links)
    workflow.add_node("weather_forecaster", weather_forecaster.weather_forecaster)
    workflow.add_node("packing_list_generator", packing_list_generator.packing_list_generator)
    workflow.add_node("food_culture_recommender", food_culture_recommender.food_culture_recommender)
    workflow.add_node("chat", chat_agent.chat_node)
    
    # Configure workflow
    workflow.set_entry_point("generate_itinerary")
    
    # Add edges (all direct to END in this simple flow)
    for node in [
        "generate_itinerary",
        "recommend_activities",
        "fetch_useful_links",
        "weather_forecaster",
        "packing_list_generator",
        "food_culture_recommender",
        "chat"
    ]:
        workflow.add_edge(node, END)
    
    return workflow.compile()

def create_initial_state(preferences: dict) -> GraphState:
    """Create initial state for the workflow."""
    preferences_text = (
        f"Destination: {preferences['destination']}\n"
        f"Month: {preferences['month']}\n"
        f"Duration: {preferences['duration']} days\n"
        f"People: {preferences['num_people']}\n"
        f"Type: {preferences['holiday_type']}\n"
        f"Budget: {preferences['budget_type']}\n"
        f"Comments: {preferences['comments']}"
    )
    
    return {
        "preferences_text": preferences_text,
        "preferences": preferences,
        "itinerary": "",
        "activity_suggestions": "",
        "useful_links": [],
        "weather_forecast": "",
        "packing_list": "",
        "food_culture_info": "",
        "chat_history": [],
        "user_question": "",
        "chat_response": ""
    }

def generate_travel_plan(preferences: dict) -> GraphState:
    """Main function to generate a travel plan using the agent workflow."""
    # Initialize components
    llm, search = initialize_agents()
    
    # Create workflow
    graph = create_workflow()
    
    # Prepare initial state
    initial_state = create_initial_state(preferences)
    
    # Execute workflow
    return graph.invoke(initial_state)

def get_user_preferences():
    return {
        "destination": input("Enter your preferred destination: "),
        "month": input("Enter the month of travel: "),
        "duration": int(input("Enter the duration of the trip (in days): ")),
        "num_people": input("Enter the number of people traveling: "),
        "holiday_type": input("Enter the type of holiday (e.g., Romantic, Adventure, Family): "),
        "budget_type": input("Enter your budget type (e.g., Budget, Mid-Range, Luxury): "),
        "comments": input("Any additional comments or preferences: ")
    }

if __name__ == "__main__":
    try:
        user_preferences = get_user_preferences()
        result = generate_travel_plan(user_preferences)
        print("\nGenerated Itinerary:")
        print(result["itinerary"])
    except Exception as e:
        print(f"Error generating travel plan: {str(e)}")

