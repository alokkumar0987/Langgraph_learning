import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from tools.search_tools import SearchTools  # <-- your custom tool

# Load keys
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize Groq LLM
llm = ChatGroq(model="llama3-8b-8192")

# Use your custom Tavily search tool
search_results = SearchTools.search_internet.invoke("bitcoin price live")

# Send Tavily results to Groq
query = f"Summarize this search result into a tweet:\n{search_results}"
resp = llm.invoke(query)

print(resp.content)
