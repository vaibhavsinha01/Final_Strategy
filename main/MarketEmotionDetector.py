import requests
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

stocks = ['Reliance']

def get_news(api_key, company_name, num_articles=100):
    # Get the date for one day ago
    start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    # News API endpoint
    endpoint = "https://newsapi.org/v2/everything"

    # Set the parameters
    params = {
        'apiKey': api_key,
        'q': company_name,
        'sortBy': 'publishedAt',
        'language': 'en',
        'pageSize': num_articles,
        'from': start_date,
        'to': datetime.now().strftime('%Y-%m-%d')
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
            all_texts = ""
            for article in articles:
                if company_name.lower() in article['title'].lower():
                    title = article['title']
                    description = article['description']
                    url = article['url']
                    
                    print(f"Title: {title}")
                    print(f"Description: {description}")
                    print(f"URL: {url}")
                    print("\n" + "="*50 + "\n")

                    # Concatenate title and description to form the text for sentiment analysis
                    all_texts += f"{title} {description} "

            return all_texts.strip()
        else:
            # Print an error message if the request was not successful
            print(f"Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Replace 'YOUR_API_KEY' with your actual News API key
api_key = '1f30615161cf4037b369adb5e4e0efb6'

# Call the function to get news about the specified company
def main():
    nltk.download('vader_lexicon')
    analyzer = SentimentIntensityAnalyzer()
    
    for company_name in stocks:
        text = get_news(api_key, company_name, num_articles=100)
        if text:
            sentiment = analyzer.polarity_scores(text)
            print(f"Text: {text}")
            print('')
            print(f"The negative Sentiment is: {sentiment['neg']}")
            print(f"The neutral Sentiment is: {sentiment['neu']}")
            print(f"The positive Sentiment is: {sentiment['pos']}")
            print(f"The compound Sentiment is: {sentiment['compound']}")

            if sentiment['pos'] > sentiment['neg']:
                print('Buy')
            elif sentiment['pos'] < sentiment['neg']:
                print('Sell')
            else:
                print('Hold')
        else:
            print(f"No relevant articles found for {company_name}")

main()
#completed 
#on 18 june showing buy for apple and sell for reliance