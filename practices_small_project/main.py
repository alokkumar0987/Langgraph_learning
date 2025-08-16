from textwrap import dedent
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from tools.search_tools import SearchTools 
from trip_tasks import TripState

from trip_agent import TripAgents
from trip_tasks import TripState

load_dotenv()


class TripGraph:
    def __init__(self, origin, cities, date_range, interests):
        self.origin = origin
        self.cities = cities
        self.date_range = date_range
        self.interests = interests

        self.agents = TripAgents()
        self.tasks = TripState()

    def build(self):
        workflow = StateGraph(state_schema=TripState)

        # Define agents
        city_selector = self.agents.city_selection_agent()
        local_expert = self.agents.local_expert()
        travel_concierge = self.agents.travel_concierge()

        # Define tasks (as nodes in graph)
        identify_task = self.tasks.identify_task(
            city_selector,
            self.origin,
            self.cities,
            self.interests,
            self.date_range
        )
        gather_task = self.tasks.gather_task(
            local_expert,
            self.origin,
            self.interests,
            self.date_range
        )
        plan_task = self.tasks.plan_task(
            travel_concierge,
            self.origin,
            self.interests,
            self.date_range
        )

        # Add nodes
        workflow.add_node("identify_city", identify_task)
        workflow.add_node("gather_info", gather_task)
        workflow.add_node("plan_trip", plan_task)

        # Define order
        workflow.add_edge("identify_city", "gather_info")
        workflow.add_edge("gather_info", "plan_trip")
        workflow.add_edge("plan_trip", END)

        workflow.set_entry_point("identify_city")

        return workflow.compile()

    def run(self):
        graph = self.build()
        user_query = HumanMessage(
            content=f"Plan a trip from {self.origin} to {self.cities} during {self.date_range} with interests in {self.interests}"
        )
        result = graph.invoke({"messages": [user_query]})
        return result


if __name__ == "__main__":
    print("## Welcome to Trip Planner (LangGraph)")
    print('---------------------------------------')

    location = input(
        dedent("""\nFrom where will you be traveling from?\n""")
    )
    cities = input(
        dedent("""\nWhat are the cities options you are interested in visiting?\n""")
    )
    date_range = input(
        dedent("""\nWhat is the date range you are interested in traveling?\n""")
    )
    interests = input(
        dedent("""\nWhat are some of your high level interests and hobbies?\n""")
    )

    trip_graph = TripGraph(location, cities, date_range, interests)
    result = trip_graph.run()

    print("\n\n########################")
    print("## Here is your Trip Plan")
    print("########################\n")
    print(result)