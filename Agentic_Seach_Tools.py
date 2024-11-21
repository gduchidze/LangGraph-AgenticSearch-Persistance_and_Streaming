from dotenv import load_dotenv
import os
from tavily import TavilyClient
import requests, re
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

_ = load_dotenv()
client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))

# result = client.search("What is Capital of Country Georgia?",
#                        include_answer=True)
#
# print(result["answer"])


city = "Rustavi"
query = f"What is the current weather in {city}?, Should i go out today? 'amindi.ge' "

ddg = DDGS()

def search(query, max_results=6):
    try:
        results = ddg.text(query, max_results=max_results)
        return [i["href"] for i in results]
    except Exception as e:
        print(f"returning previous results due to exception reaching ddg.")
        results = [
            "https://weather.com/weather/today/l/USCA0987:1:US",
            "https://weather.com/weather/hourbyhour/l/54f9d8baac32496f6b5497b4bf7a277c3e2e6cc5625de69680e6169e7e38e9a8",
        ]
        return results


for i in search(query):
    print(i)


def scrape_weather_info(url):
    """Scrape content from the given URL"""
    if not url:
        return "Weather information could not be found."

    # fetch data
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return "Failed to retrieve the webpage."

    # parse result
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup

url = search(query)[0]

# scrape first wesbsite
soup = scrape_weather_info(url)
#
# print(f"Website: {url}\n\n")
# print(str(soup.body)[:50000]) # limit long outputs

# extract text
weather_data = []
for tag in soup.find_all(['h1', 'h2', 'h3', 'p']):
    text = tag.get_text(" ", strip=True)
    weather_data.append(text)

# combine all elements into a single string
weather_data = "\n".join(weather_data)

# remove all spaces from the combined text
weather_data = re.sub(r'\s+', ' ', weather_data)

# print(f"Website: {url}\n\n")
# print(weather_data)

# run search
result = client.search(query, max_results=1)

# print first result
data = result["results"][0]["content"]

print(data)


import json
from pygments import highlight, lexers, formatters

# parse JSON
parsed_json = json.loads(data.replace("'", '"'))

# pretty print JSON with syntax highlighting
formatted_json = json.dumps(parsed_json, indent=4)
colorful_json = highlight(formatted_json,
                          lexers.JsonLexer(),
                          formatters.TerminalFormatter())

print(colorful_json)

memory = SqliteSaver.from_conn_string(":memory:")

