from textwrap import dedent
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from tools.search_tools import SearchTools 




from tools.search_tools import SearchTools 
from trip_tasks import TripState, TripTasks  # Add TripTasks here
from trip_agent import TripAgents






load_dotenv()


class TripGraph:
    def __init__(self, origin, cities, date_range, interests):
        self.origin = origin
        self.cities = [c.strip() for c in cities.split(",")]
        self.date_range = date_range
        self.interests = [i.strip() for i in interests.split(",")]

        self.agents = TripAgents()
        self.tasks = TripTasks(self.agents)

    def build(self):
        workflow = StateGraph(TripState)

        # Add nodes
        workflow.add_node("identify_city", self.tasks.identify_task())
        workflow.add_node("gather_info", self.tasks.gather_task())
        workflow.add_node("plan_trip", self.tasks.plan_task())

        # Define order
        workflow.add_edge("identify_city", "gather_info")
        workflow.add_edge("gather_info", "plan_trip")
        workflow.add_edge("plan_trip", END)

        workflow.set_entry_point("identify_city")
        return workflow.compile()

    def run(self):
        graph = self.build()
        initial_state = TripState(
            origin=self.origin,
            cities=self.cities,
            interests=self.interests,
            trip_range=self.date_range
        )
        result = graph.invoke(initial_state)
        return result["result"]


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