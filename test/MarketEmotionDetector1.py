import requests
from datetime import datetime, timedelta

stocks=['Bajaj Finance']

def get_news(api_key, company_name, num_articles=100, days_ago=1):
    start_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')

    # News API endpoint
    endpoint = "https://newsapi.org/v2/everything"

    # Set the parameters
    params = {
        'apiKey': api_key,
        'q': company_name,
        'sortBy': 'publishedAt',
        'language': 'en',
        'pageSize': num_articles,
        'from': start_date
    }

    try:
        # Make the request to the API
        response = requests.get(endpoint, params=params)

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse the JSON response
            data = response.json()

            # Extract and print the news articles
            articles = data['articles']
            for article in articles:
                if company_name.lower() in article['title'].lower():
                    print(f"Title: {article['title']}")
                    print(f"Description: {article['description']}")
                    print(f"URL: {article['url']}")
                    print("\n" + "="*50 + "\n")
        else:
            # Print an error message if the request was not successful
            print(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Replace 'YOUR_API_KEY' with your actual News API key
api_key = '1f30615161cf4037b369adb5e4e0efb6'

# Call the function to get news about the specified company
def main():
    for company_name in stocks:
        get_news(api_key, company_name, num_articles=100, days_ago=10)
main()