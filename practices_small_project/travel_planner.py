# travel_planner.py
import os
import requests
from textwrap import dedent
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from typing import TypedDict, List, Optional
from langchain.tools import tool

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ====================== Search Tools ====================== #
class SearchTools:
    @tool("Search the internet with Tavily")
    def search_internet(query: str):
        """Search the internet about a given topic and return relevant results"""
        top_result_to_return = 5
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "num_results": top_result_to_return
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code != 200:
            return f"Error: {response.text}"
        
        results = response.json().get("results", [])
        if not results:
            return "No results found. Try a different query."
        
        return "\n\n".join([
            f"Title: {result['title']}\nLink: {result['url']}\nSnippet: {result['content']}\n"
            for result in results[:top_result_to_return]
       ] )

# ====================== State Definition ====================== #
class TripState(TypedDict):
    origin: str
    cities: List[str]
    interests: List[str]
    date_range: str
    selected_city: Optional[str]
    city_guide: Optional[str]
    itinerary: Optional[str]
    search_results: Optional[str]

# ====================== Core Agents ====================== #
class TripPlanner:
    def __init__(self):
        self.llm = ChatGroq(model="llama3-8b-8192", api_key=GROQ_API_KEY)
        self.search_tool = SearchTools()
    
    def city_selector(self, state: TripState) -> dict:
        """Select best city based on preferences"""
        query = dedent(f"""
        Select the best city for a trip from {state['origin']} considering:
        - Traveler interests: {", ".join(state['interests'])}
        - City options: {", ".join(state['cities'])}
        - Travel dates: {state['date_range']}
        
        Analyze based on:
        - Weather during travel period
        - Seasonal events/festivals
        - Travel costs and availability
        - Relevance to traveler interests
        """)
        resp = self.llm.invoke(query)
        return {"selected_city": resp.content.strip()}
    
    def city_researcher(self, state: TripState) -> dict:
        """Research city using Tavily and generate insights"""
        if not state.get('selected_city'):
            raise ValueError("No city selected for research")
        
        # Perform Tavily search
        search_query = dedent(f"""
        Comprehensive travel guide for {state['selected_city']} including:
        - Top attractions and hidden gems
        - Cultural experiences during {state['date_range']}
        - Local customs and etiquette
        - Weather in {state['date_range'].split('-')[0].strip()}
        - Cost of living and travel expenses
        - Safety considerations
        - Recommendations for: {", ".join(state['interests'])}
        """)
        search_results = self.search_tool.search_internet.invoke(search_query)
        
        # Generate city guide
        guide_prompt = dedent(f"""
        Create a comprehensive travel guide for {state['selected_city']}:
        
        Traveler Profile:
        - Interests: {", ".join(state['interests'])}
        - Dates: {state['date_range']}
        - Origin: {state['origin']}
        
        Research Findings:
        {search_results}
        
        Structure your guide with:
        1. Introduction to the city
        2. Top attractions with descriptions
        3. Hidden gems
        4. Cultural highlights during travel dates
        5. Dining recommendations
        6. Accommodation options
        7. Transportation tips
        8. Budget information
        9. Safety and etiquette
        10. Packing suggestions
        """)
        resp = self.llm.invoke(guide_prompt)
        return {
            "city_guide": resp.content.strip(),
            "search_results": search_results
        }
    
    def itinerary_creator(self, state: TripState) -> dict:
        """Create detailed itinerary using city guide"""
        if not state.get('selected_city'):
            raise ValueError("No city selected for itinerary")
        if not state.get('city_guide'):
            raise ValueError("No city guide available")
        
        query = dedent(f"""
        Create a detailed travel itinerary for {state['selected_city']}:
        
        Traveler Details:
        - Origin: {state['origin']}
        - Dates: {state['date_range']}
        - Interests: {", ".join(state['interests'])}
        
        City Guide Insights:
        {state['city_guide']}
        
        Itinerary Requirements:
        - Cover full duration of stay
        - Include morning/afternoon/evening activities
        - Restaurant recommendations
        - Hotel suggestions
        - Transportation between locations
        - Budget estimates
        - Packing list
        - Weather considerations
        - Backup options
        """)
        resp = self.llm.invoke(query)
        return {"itinerary": resp.content.strip()}

# ====================== Workflow Graph ====================== #
def build_travel_planner():
    planner = TripPlanner()
    
    # Create state graph
    workflow = StateGraph(TripState)
    
    # Add nodes
    workflow.add_node("select_city", planner.city_selector)
    workflow.add_node("research_city", planner.city_researcher)
    workflow.add_node("create_itinerary", planner.itinerary_creator)
    
    # Define flow
    workflow.add_edge("select_city", "research_city")
    workflow.add_edge("research_city", "create_itinerary")
    workflow.add_edge("create_itinerary", END)
    
    workflow.set_entry_point("select_city")
    return workflow.compile()

# ====================== User Interface ====================== #
class TravelPlannerApp:
    def __init__(self):
        self.planner = build_travel_planner()
    
    def run(self):
        print("\n===== Travel Planner Assistant =====")
        print("Please provide your trip details:\n")
        
        origin = input("Traveling from: ")
        cities = input("Cities considering (comma-separated): ").split(",")
        date_range = input("Travel dates (e.g., June 15-30, 2024): ")
        interests = input("Your interests (comma-separated): ").split(",")
        
        # Prepare initial state
        state: TripState = {
            "origin": origin.strip(),
            "cities": [c.strip() for c in cities],
            "interests": [i.strip() for i in interests],
            "date_range": date_range.strip(),
            "selected_city": None,
            "city_guide": None,
            "itinerary": None,
            "search_results": None
        }
        
        # Execute planning workflow
        result = self.planner.invoke(state)
        
        # Display results
        print("\n\n===== TRAVEL PLAN =====")
        print(f"Selected City: {result['selected_city']}")
        print(f"\n===== CITY GUIDE =====")
        print(result['city_guide'])
        print(f"\n===== DETAILED ITINERARY =====")
        print(result['itinerary'])

# ====================== Main Execution ====================== #
if __name__ == "__main__":
    app = TravelPlannerApp()
    app.run()