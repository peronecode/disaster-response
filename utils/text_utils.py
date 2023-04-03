import re
import string
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract whether the text starts with a verb.
    """

    def starting_verb(self, text):
        """
        Check if the input text starts with a verb.
        
        Args:
        text: str, input text.
        
        Returns:
        bool, True if the text starts with a verb, False otherwise.
        """
        for sentence in nltk.sent_tokenize(text):
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags) > 0:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        """
        Transform the input data by applying the 'starting_verb' method.
        
        Args:
        X: DataFrame or Series, input data.
        
        Returns:
        DataFrame, transformed data with an additional column.
        """
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


stop_words = stopwords.words('english') + list(string.punctuation)
lemmatizer = WordNetLemmatizer()
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
def tokenize(text):
    """
    Tokenize and clean text by removing URLs, stop words, and applying lemmatization.
    
    Args:
    text: str, input text.
    
    Returns:
    clean_tokens: list, tokenized and cleaned text.
    """
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize to lower case and strip the token
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens if tok.lower() not in stop_words]

    return clean_tokens