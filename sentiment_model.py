import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords

df = pd.read_csv('coursesentiment.csv', index_col = 0)
df_neg = df[df.Positive == 0]
df_pos = df[df.Positive == 1]
df_pos = df_pos.sample(int(1.5 * len(df_neg)), random_state = 24)
df = pd.concat([df_neg, df_pos], ignore_index = True)

stop = set(stopwords.words('english'))

def text_preprocessing(data):
    '''Removes punctuation, digits and underscores from text. Returns the tokenized version in lowercase'''
    data = data.lower()
    data = re.sub(r'[_\',\.]+', ' ', data)
    data = re.sub(r'not\s*(?:\w+\s*)?bad', 'good', data)
    data = re.sub('\d+', ' ', data)
    words = [word for word in nltk.word_tokenize(data)]
    data = ' '.join(words)
    return data

df['Review'] = df['Review'].apply(text_preprocessing)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Positive'], test_size = 0.1, random_state = 24)

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df = 5, ngram_range = (1,3)).fit(X_train)
X_train_vectorized = vect.transform(X_train)
X_test_vectorized = vect.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

def model_pred(model):
    model.fit(X_train_vectorized, y_train)
    pred = model.predict(X_test_vectorized)
    return f1_score(y_test, pred), roc_auc_score(y_test, pred)

lr_model = LogisticRegression(solver = 'saga', C = 0.1, max_iter = 3000)
model_pred(lr_model)

def make_prediction(model, input_text: str, proba=False) -> float:
    text = text_preprocessing(input_text)
    if proba == True:
        return model.predict_proba(vect.transform([text]))[:,1][0]
    return model.predict(vect.transform([text]))

with open('lr_model.pkl', 'wb') as f:
    pickle.dump(lr_model, f)

with open('vect.pkl', 'wb') as g:
    pickle.dump(vect, g)