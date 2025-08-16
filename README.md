mm m# Langgraph_learning

# Installation
```
%pip install -qU langchain-groq python-dotenv


#method 1. Direct access API key 
```
import os

os.environ["GROQ_API_KEY"] = "your_api_key_here"

```
#Method 2. through: .env using 
```


GROQ_API_KEY=your_secret_api_key_here

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key
API_KEY = os.getenv("GROQ_API_KEY")

print("API Key loaded:", API_KEY[:5] + "*****")  # mask for safety

```

#Use api key request 
```
import requests

url = "https://api.groq.com/v1/some-endpoint"
headers = {"Authorization": f"Bearer {API_KEY}"}

response = requests.get(url, headers=headers)
print(response.json())

```

# USING GROQ_API_KAY

```
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=None,
    reasoning_format="parsed",
    timeout=None,
    max_retries=2,
    # other params...
)


messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)
```
