mm m# Langgraph_learning



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
