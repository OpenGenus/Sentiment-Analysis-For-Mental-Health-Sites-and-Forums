from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_score = analyzer.polarity_scores(text)['compound']

    if sentiment_score >= 0.05:
        return 'Positive'
    elif sentiment_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def get_text(url):
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return text
    else:
        print(f"Failed to retrieve content from {url}. Status code: {response.status_code}")
        return None

def prepare_dataset(url):
    text = get_text(url)
    if text:
        words = text.split()
        sentiments = [analyze_sentiment(word) for word in words]
        df = pd.DataFrame({'Word': words, 'Sentiment': sentiments})
        df.to_csv('website_sentiments.csv', index=False)

def visualize_sentiments():
    df = pd.read_csv('website_sentiments.csv')

    if 'Word' not in df.columns or 'Sentiment' not in df.columns:
        print("Dataset columns are not as expected. Please check the dataset format.")
        return

    sentiment_counts = df['Sentiment'].value_counts()

    plt.figure(figsize=(8, 6))
    sb.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis")
    plt.title('Number of Neutral, Positive, and Negative Words')
    plt.xlabel('Sentiment')
    plt.ylabel('Word Count')
    plt.show()

website_url = "https://manochikitsa.com/topic/i-feeling-anxiety-and-dont-feel-like-doing-anything/"

prepare_dataset(website_url)
visualize_sentiments()
