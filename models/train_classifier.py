# import libraries
import sys
import pandas as pd 
import re
from sqlalchemy import create_engine
import nltk
nltk.download(['stopwords','punkt','wordnet'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle


def load_data(database_filepath):
    """
    Loads data from SQLite database.
    
    Parameters:
    database_filepath: Filepath to the database
    
    Input: sqlite database file
    Output: X feature and y target variables
    """
    
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('disaster_response_table', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    
    return X,Y


def tokenize(text):
    """
    Tokenizes and lemmatizes text.
    
    Parameters:
    text: Text to be tokenized
    
    Input: text to be tokenize and lemmatize
    Output: list of clean token
    """
    clean_text = re.sub('[^a-zA-Z0-9]',' ',text) # Leave out any characters from A-Z, a-z, or 0-9
    
    clean_text = clean_text.lower() # Make every character lowercase.
    
    clean_words = word_tokenize(clean_text) # Text is tokenized into a list of clean_words.
    
    topwords = stopwords.words("english")
    clean_words = [w for w in clean_words if not w in topwords] # Remove all English stopwords
    
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in clean_words] # Lemmatize words
    
    return lemmed

def build_model():
    """
    Builds classifier and tunes model using GridSearchCV.
    
    Input: None
    Output: best model
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])
    parameters = {
        'clf__estimator__n_estimators' : [1]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    return cv


def evaluate_model(model, X_test, Y_test):
    """
    Evaluates the performance of model and returns classification report. 
    
    Parameters:
    model: classifier
    X_test: test dataset
    Y_test: labels for test data in X_test
    
    Input: model to evaluate, data input to test and label of them
    Output: classification reports of each columns (precision, recall, f1-score, accuracy)
    """
    y_pred = model.predict(X_test)
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))


def save_model(model, model_filepath):
    """
    The finished model is exported as a pickle file.
    
    Input: model to save, filepath of the saved model 
    Output: None
    """
    model_name = model_filepath
    pickle.dump(model, open(model_name, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()