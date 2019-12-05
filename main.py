from collections import defaultdict

import nltk
import os
import numpy as np

import sklearn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from num2words import num2words
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stopWords = set(stopwords.words('english'))


# region Prepocessing


def convert_lower_case(data):
    return np.char.lower(data)


def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))
    new_text = ""
    for w in words:
        if w not in stop_words and len(w) > 1:
            new_text = new_text + " " + w
    return new_text


def remove_punctuation(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data


def remove_apostrophe(data):
    return np.char.replace(data, "'", "")


def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text


def lemmanize(data):
    lemmatizer = nltk.WordNetLemmatizer()

    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + lemmatizer.lemmatize(w)
    return new_text


def preprocess(data):
    data = convert_lower_case(data)
    data = remove_punctuation(data)
    data = remove_apostrophe(data)
    data = remove_stop_words(data)
    data = lemmanize(data)
    return data

# endregion


token_dict = defaultdict(list)
path = "./categories"

data_array = []
target_array = []

print('Starting document categorizer... \n')

print('Loading files... \n')

for dirpath, dirs, files in os.walk(path):
    striped_path = dirpath.replace('./', '')
    if striped_path.startswith('.') or not striped_path or striped_path == 'venv':
        continue
    for f in files:
        fname = os.path.join(dirpath, f)
        dname = os.path.dirname(fname)
        category_name = os.path.basename(dname)
        if not fname.endswith('.txt'):
            continue
        with open(fname) as pearl:
            text = pearl.read()
            data_array.append(text)
            target_array.append(category_name)

print('Preprocessing data... \n')

processed_data_array = [preprocess(data) for data in data_array]
data_bunch = sklearn.datasets.base.Bunch(data=processed_data_array, target=target_array)

print('Vectorizing data... \n')

tfidfconverter = TfidfVectorizer(max_features=1500, max_df=.65, min_df=1, stop_words='english', use_idf=True, norm=None)
X = tfidfconverter.fit_transform(data_bunch.data).toarray()

print('Splitting data... \n')

X_train, X_test, y_train, y_test = train_test_split(X, data_bunch.target, test_size=0.5, random_state=0)

print('Training classifier... \n')

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)

print('Predicting outputs... \n')

y_pred = classifier.predict(X_test)

print('Finished! Here are the results: \n')

print('Confusion matrix: \n')
print(confusion_matrix(y_test, y_pred))
print('\nClassification report: \n')
print(classification_report(y_test, y_pred))
print('\nAccuracy score: \n')
print(accuracy_score(y_test, y_pred))
