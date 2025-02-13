from flask import Flask, jsonify
from flask_cors import CORS
import stockSentiment

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

@app.route('/<ticker>')
def getSentiment(ticker):
    articles = stockSentiment.getNewsData(ticker)
    return jsonify({
        "sentiment": stockSentiment.predict_sentiment(articles), 
        "articles": stockSentiment.get_articles()
    })

if __name__ == '__main__':
        app.run(debug=True, port=5000)