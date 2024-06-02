import re
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    text = text.lower()  # Lowercasing
    text = re.sub(r'\d+', '', text)  # Removing numbers
    text = re.sub(r'[^\w\s]', '', text)  # Removing punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Removing extra whitespace
    return text


def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens


def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens

def preprocess_text(df):
    df['cleaned_text'] = df['Article_content'].apply(clean_text)
    df['tokens'] = df['cleaned_text'].apply(tokenize_text)
    df['filtered_tokens'] = df['tokens'].apply(remove_stopwords)
    df['lemmatized_tokens'] = df['filtered_tokens'].apply(lemmatize_tokens)
    df['final_text'] = df['lemmatized_tokens'].apply(lambda tokens: ' '.join(tokens))
    return df

