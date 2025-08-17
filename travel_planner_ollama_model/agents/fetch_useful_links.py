from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv

load_dotenv()

def fetch_useful_links(state):
    # Tavily tool initialize
    search = TavilySearchResults(k=5)

    destination = state['preferences'].get('destination', '')
    month = state['preferences'].get('month', '')
    query = f"Travel tips and guides for {destination} in {month}"

    try:
        # Tavily API se results fetch karo
        search_results = search.invoke(query)

        # Agar Tavily se dict list return hota hai
        links = [
            {"title": result.get("title", "No title"), "link": result.get("url", "")}
            for result in search_results
        ][:5]

        return {"useful_links": links}

    except Exception as e:
        return {"useful_links": [], "warning": f"Failed to fetch links: {str(e)}"}