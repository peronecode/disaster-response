import sys
import re
import string

from sqlalchemy import create_engine

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils.fixes import loguniform

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


def load_data(database_filepath):
    """
    Load data from the SQLite database.
    
    Args:
    database_filepath: str, path to the SQLite database file.
    
    Returns:
    X: DataFrame, feature variables (messages).
    y: DataFrame, target variables (categories).
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Messages', engine, index_col='index').sample(800)

    X = df['message']
    y = df.drop(['id', 'original', 'message', 'genre'], axis=1)

    return X, y


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


def build_model():
    """
    Build the model using a pipeline and grid search.
    Returns:
    cv: GridSearchCV, the grid search object containing the model.
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('starting_verb', StartingVerbExtractor())
        ])),
        ("clf", MultiOutputClassifier(RandomForestClassifier())),
    ], verbose=True)

    rf_params = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator': [RandomForestClassifier()],
        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3, 4]
    }

    gb_params = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator': [OneVsRestClassifier(GradientBoostingClassifier())],
        'clf__estimator__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__estimator__min_samples_split': [2, 5, 10],
        'clf__estimator__estimator__learning_rate': [0.01, 0.1, 1],
    }

    svc_params = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator': [OneVsRestClassifier(SVC())],
        'clf__estimator__estimator__C': [0.1, 1, 10],
        'clf__estimator__estimator__gamma': [0.1, 1, 10],
        'clf__estimator__estimator__kernel': ['linear', 'rbf', 'poly'],
    }

    nb_params = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'clf__estimator': [MultinomialNB()],
        'clf__estimator__alpha': [0.01, 0.1, 1],
    }

    cv_params = [rf_params, gb_params, svc_params, nb_params]
    return GridSearchCV(pipeline, param_grid=cv_params)


def evaluate_model(model, X_test, Y_test):
    """
    Evaluate the model by calculating various performance metrics.
    Args:
    model: GridSearchCV, the trained model.
    X_test: DataFrame, test data.
    Y_test: DataFrame, test labels.
    """
    y_pred = model.predict(X_test)

    for i, col in enumerate(Y_test.columns):
        col_weighted_avg = classification_report(Y_test.iloc[:, i], y_pred[:, i], output_dict=True)['weighted avg']
        f1_score = col_weighted_avg['f1-score']
        precision = col_weighted_avg['precision']
        recall = col_weighted_avg['recall']
        print(f'{col}\nF1-score:{f1_score:.2f} / Precision:{precision:.2f} / Recall:{recall:.2f}\n')


def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file.

    Args:
    model: GridSearchCV, the trained model.
    model_filepath: str, path to save the pickle file.
    """
    pd.to_pickle(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data...\n DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
        
    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()