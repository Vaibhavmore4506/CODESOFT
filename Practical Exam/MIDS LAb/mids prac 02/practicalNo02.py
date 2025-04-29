# Import Required Libraries
import pandas as pd
import string
import nltk

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

# Download necessary NLTK data files (only once)
nltk.download('stopwords')
nltk.download('punkt')

# Load Dataset
df = pd.read_csv("Tweets.csv")

# Feature Selection
df = df.iloc[:, [10, 1]]  # Select only 'text' and 'airline_sentiment'
df.columns = ['text', 'airline_sentiment']  # Rename columns properly

# Print Before Preprocessing
print("Before Preprocessing")
print(df[['text', 'airline_sentiment']].head())
print("\n")

# Preprocessing: Remove Stopwords & Punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words('english'))

def clean_text(text):
    tokens = word_tokenize(str(text))  # Convert to string in case of NaN
    cleaned_text = [word for word in tokens if word.lower() not in stop_words and word not in string.punctuation]
    return ' '.join(cleaned_text)

# Apply Cleaning
df['text'] = df['text'].apply(clean_text)

# Print After Cleaning
print("After Cleaning of Text Column")
print(df[['text', 'airline_sentiment']].head())
print("\n")

# Feature Extraction: TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])

# Target variable
y = df['airline_sentiment']

# Print After Preprocessing
print("After Preprocessing")
print(df[['text', 'airline_sentiment']].head())
print("\n")

# Model Training: K-Nearest Neighbors Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

# Print After Model Training
print("After Model Training")
print(df[['text', 'airline_sentiment']].head())
print("\n")

# Predict Sentiment for Unknown Reviews
def predict_sentiment(review):
    cleaned_review = clean_text(review)
    review_vectorized = vectorizer.transform([cleaned_review])
    prediction = knn.predict(review_vectorized)
    return prediction[0]

# Example usage:
unknown_review = "The product was amazing, I really loved it!"
predicted_sentiment = predict_sentiment(unknown_review)

print(f"Input Review: {unknown_review}")
print(f"Predicted Sentiment: {predicted_sentiment}")
