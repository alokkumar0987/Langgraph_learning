import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

load_dotenv()
# Initialize LLM
llm = ChatGroq(model="llama3-8b-8192")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
from pydantic import BaseModel

class TripState(BaseModel):
    origin: str
    cities: list
    interests: list
    trip_range: str
    result: str = ""

class TripTasks:
    def __init__(self, agents):
        self.agents = agents

    def identify_task(self):
        return self.agents.city_selection_agent()

    def gather_task(self):
        return self.agents.local_expert()

    def plan_task(self):
        return self.agents.travel_concierge()


