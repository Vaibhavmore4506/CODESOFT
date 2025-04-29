import pandas as pd 
import matplotlib.pyplot as plt 
# Step 1: Define positive and negative word lists 
positive_words = ["good", "great", "happy", "love", "excellent", "fantastic", "awesome", "pleased", "enjoy"] 
negative_words = ["bad", "terrible", "sad", "hate", "poor", "awful", "worst", "angry", "disappointed"]
 # Step 2: Read CSV file 
df = pd.read_csv("reviews.csv") # Make sure 'reviews.csv' has a column named 'text' 
 # # Step 3: Define sentiment classifier
def classify_sentiment(text): 
    tokens = text.lower().split() 
    pos = sum(1 for word in tokens if word in positive_words) 
    neg = sum(1 for word in tokens if word in negative_words)
     #11th 12th alternative method/explanation #for word in tokens: #if word in positive_words: # positive_count += 1 #elif word in negative_words: # negative_count += 1
    if pos > neg:
         return "Positive" 
    elif neg > pos: 
        return "Negative" 
    else: return "Neutral" 
    # Step 4: Apply classifier to each document 
df['Sentiment'] = df['text'].apply(classify_sentiment)
print(df[['text', 'Sentiment']]) 
# Step 6: Plot sentiment distribution 
sentiment_counts = df['Sentiment'].value_counts()
plt.figure(figsize=(6, 4)) 
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray']) 
plt.title("Sentiment Analysis Result") 
plt.xlabel("Sentiment") 
plt.ylabel("Number of Documents") 
plt.xticks(rotation=0) 
plt.grid(axis='y', linestyle='--', alpha=0.7) 
plt.tight_layout()
plt.show()