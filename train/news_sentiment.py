import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from collections import Counter

# Load Financial PhraseBank dataset (assuming Sentence_50agree.csv is available)
df = pd.read_csv("output.csv", encoding="ISO-8859-1")

# Rename columns (assuming first column is sentences and second is sentiment labels)
df.columns = ["text", "sentiment"]

# Map sentiment labels to integers
sentiment_mapping = {"negative": 0, "neutral": 1, "positive": 2}
df["sentiment"] = df["sentiment"].map(sentiment_mapping)

# Split into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(), df["sentiment"].tolist(), test_size=0.2, random_state=42
)

# Load DistilBERT tokenizer 
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the dataset
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

# Convert to Dataset
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = SentimentDataset(train_encodings, train_labels)
test_dataset = SentimentDataset(test_encodings, test_labels)

# Load Pretrained DistilBERT Model (3 sentiment classes: Negative, Neutral, Positive)
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,  # Increased to 10 for better accuracy
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Custom function to move batches to the correct device
def collate_fn(batch):
    return {key: torch.stack([b[key] for b in batch]) for key in batch[0]}

# Trainer API (for fine-tuning the model)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=collate_fn,  # Ensures data is moved correctly
)

# Train the model
trainer.train()

# Save the trained model
model.save_pretrained("fine-tuned-stock-sentiment")

def predict_sentiment(articles):
    inputs = tokenizer(articles, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Move inputs to same device as model
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=1).tolist()
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    
     # Get the predicted sentiment labels
    predicted_labels = [sentiment_labels[pred] for pred in predictions]
    
    # Calculate the most frequent sentiment
    most_common_sentiment = Counter(predicted_labels).most_common(1)[0][0]
    
    return most_common_sentiment


# Example Usage
articles = [
    "AMD Stock Plummets as Investors Lose Confidence in Growth Prospects",
    "AMD Struggles to Compete with NVIDIA’s AI Dominance, Market Share Slips",
    "AMD’s Latest Chips Fail to Impress, Performance Falls Short of Expectations",
    "NVIDIA’s RTX 5090 launch could be the worst ever — can AMD capitalize?",
    "Insider Selling Raises Concerns About AMD’s Long-Term Prospects",
    "AMD might’ve already lost the war with the RX 9070 XT.",
    "Earnings Miss Sends AMD Shares Tumbling, Analysts Cut Price Targets",
    "3 problems for the stock price of Nvidia rival AMD. None",
    "Investors worry DeepSeek reduces AI chip demand, but there's a case for remaining bullish on Nvidia.",
    "Where to buy the RTX 5090 and RTX 5080 today.",
    "Call of Duty: Black Ops 6 and Warzone causing blue screen errors on PC.",
    "3 GPUs you should buy instead of the RTX 5080.",
    "AMD's New Radeon 9 9070 Series Graphics Cards Launch Soon, Support FSR4 Upscaling.",
    "RTX 5080 benchmarks bode well for Radeon RDNA 4 GPUs — AMD's opportunity has never been better.",
    "What's Happening With AMD Stock? AMD stock (NASDAQ:AMD) has seen a meaningful sell-off over the past week."
]

final_sentiment = predict_sentiment(articles)

print(final_sentiment)

torch.save(model.state_dict(), "fin_senti.pth")
tokenizer.save_pretrained("fine-tuned-stock-sentiment-tokenizer")  

