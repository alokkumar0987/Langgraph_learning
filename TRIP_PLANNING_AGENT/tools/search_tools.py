import json
import os
import requests
from langchain.tools import tool


class SearchTools():

    @tool("Search the internet with Tavily")
    def search_internet(query: str):
        """Useful to search the internet about a given topic and return relevant results"""
        top_result_to_return = 4
        url = "https://api.tavily.com/search"

        payload = {
            "api_key": os.environ["TAVILY_API_KEY"],
            "query": query,
            "num_results": top_result_to_return
        }

        headers = {"Content-Type": "application/json"}

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code != 200:
            return f"Error: {response.text}"

        results = response.json().get("results", [])
        if not results:
            return "Sorry, I couldn't find anything. There might be an issue with your Tavily API key or query."

        string = []
        for result in results[:top_result_to_return]:
            try:
                string.append('\n'.join([
                    f"Title: {result['title']}",
                    f"Link: {result['url']}",
                    f"Snippet: {result['content']}",
                    "\n-----------------"
                ]))
            except KeyError:
                continue

        return '\n'.join(string)