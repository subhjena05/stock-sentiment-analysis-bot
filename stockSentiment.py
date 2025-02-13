from newsapi import NewsApiClient
from datetime import datetime, timedelta
import joblib
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from collections import Counter


# Reinitialize the model architecture
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

# Load the trained model weights
model.load_state_dict(torch.load("fin_senti.pth", map_location=torch.device('cpu')))  # Ensure weights are loaded onto CPU

# Load the tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("fine-tuned-stock-sentiment-tokenizer")

# Force the device to CPU
device = torch.device("cpu")  # This forces the model to use CPU
model.to(device)

# Set model to evaluation mode for inference
model.eval()

#Create the predict_sentiment function
def predict_sentiment(articles):
    inputs = tokenizer(articles, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Move inputs to same device as model
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=1).tolist()
    sentiment_labels = ["Negative", "Neutral", "Positive"]

    predicted_labels = [sentiment_labels[pred] for pred in predictions]

    most_common_sentiment = Counter(predicted_labels).most_common(1)[0][0]

    return most_common_sentiment
    
def get_articles():
    return article_list
    
#This part is just the same as the newsAPI script however at the end it feeds the information into the model
API_KEY = "caa867d4940444268bd55423be0ee3a2"
newsapi = NewsApiClient(api_key=API_KEY)

today = datetime.today()
last_week = today - timedelta(days=30)

article_list = []


#Utility function that the flask script can interact with in order to more easily get stock sentiment
def getNewsData(ticker):
    global article_list

    article_list=[]

    response = newsapi.get_everything(
    q=ticker + " stock",
    language='en',
    sort_by='relevancy',
    from_param=last_week,
    to=today,
    )

    articles = response.get('articles', [])

    if articles:
        for article in articles[:15]:  # Display top results
            article_text = f"{article['title']} - {article['description']}\n"  
            article_list.append(article_text)  # Append the formatted string to the list
    else:
        print("No articles found.")

    return article_list
