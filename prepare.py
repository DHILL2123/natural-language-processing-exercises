import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import pandas as pd

import acquire


def basic_clean(text):
    """
    This function takes in a string and applies some basic text cleaning to it:
    * Lowercase everything
    * Normalize unicode characters
    * Replace anything that is not a letter, number, whitespace or a single quote.
    """
    # Lowercase the text
    text = text.lower()
    
    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    
    # Replace everything that is not a letter, number, whitespace or a single quote with a space
    text = re.sub(r"[^a-z0-9\s']", ' ', text)
    
    # Remove extra whitespaces
    text = re.sub(r"\s+", ' ', text).strip()
    
    return text


def tokenize(text):
    """
    This function takes in a string and tokenizes all the words in the string.
    """
    # Tokenize the text using the nltk library
    tokens = nltk.word_tokenize(text)
    
    return tokens

def stem(text):
    """
    This function takes in a string and returns the text after applying stemming to all the words.
    """
    # Tokenize the text using the nltk library
    tokens = nltk.word_tokenize(text)
    
    # Apply stemming to each token using the PorterStemmer from nltk
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    
    # Join the stemmed tokens back into a single string
    stemmed_text = ' '.join(stemmed_tokens)
    
    return stemmed_text

def lemmatize(text):
    """
    This function takes in a string and returns the text after applying lemmatization to each word.
    """
    # Tokenize the text using the nltk library
    tokens = nltk.word_tokenize(text)
    
    # Apply lemmatization to each token using the WordNetLemmatizer from nltk
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join the lemmatized tokens back into a single string
    lemmatized_text = ' '.join(lemmatized_tokens)
    
    return lemmatized_text

def remove_stopwords(text, extra_words=None, exclude_words=None):
    """
    This function takes in a string and returns the text after removing all the stopwords.
    It has two optional parameters: extra_words and exclude_words to define additional stop words to include
    and words that we don't want to remove.
    """
    # Define the list of stopwords from the nltk library
    stopword_list = stopwords.words('english')
    
    # Add any extra stop words to the list
    if extra_words:
        stopword_list.extend(extra_words)
    
    # Exclude any words from the stopword list
    if exclude_words:
        stopword_list = [word for word in stopword_list if word not in exclude_words]
    
    # Tokenize the text using the nltk library
    tokens = nltk.word_tokenize(text)
    
    # Remove the stop words from the tokens
    filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    
    # Join the filtered tokens back into a single string
    filtered_text = ' '.join(filtered_tokens)
    
    return filtered_text
