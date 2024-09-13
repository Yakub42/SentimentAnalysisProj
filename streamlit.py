import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re, nltk

nltk.download('punkt_tab')

with open('lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('vect.pkl', 'rb') as g:
    vect = pickle.load(g)

def text_preprocessing(data):
    '''Removes punctuation, digits and underscores from text. Returns the tokenized version in lowercase'''
    data = data.lower()
    data = re.sub(r'[_\',\.]+', ' ', data)
    data = re.sub(r'not\s*(?:\w+\s*)?bad', 'good', data)
    data = re.sub('\d+', ' ', data)
    words = [word for word in nltk.word_tokenize(data)]
    data = ' '.join(words)
    return data

def make_prediction(model, input_text: str, proba=False) -> float:
    text = text_preprocessing(input_text)
    if proba == True:
        return model.predict_proba(vect.transform([text]))[:,1][0]
    return model.predict(vect.transform([text]))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']*20
st.write("## Welcome to course sentiment analyser")

rated_courses = pd.read_csv('ratedCourses.csv', index_col = 0)

course = st.text_input("What course would you like to rate?", max_chars = 12)
user_input2 = st.text_input(f"What did you think about {course}", max_chars = 100)
st.write('\n')
button1 = st.button("Click here for the model to provide your sentiment.")
sentiment = make_prediction(lr_model, user_input2, True)
sentiment = round(5 * sentiment, 1)

if sentiment > 3:
    str_sentiment = 'Glad you liked the course!'
else:   str_sentiment = 'Thanks for the feedback, we will use it to improve the course'
st.write('\n')
if button1: 
    st.write(str_sentiment)
    rated_courses = pd.concat([pd.DataFrame([[course, sentiment]], columns = rated_courses.columns), rated_courses], ignore_index = True) 
    st.write("Your rating has been recorded, feel free to rate another course.")
    st.dataframe(rated_courses, hide_index=True)
    avg_ratings = rated_courses.groupby('Course').agg(['mean'])
    avg_ratings.columns = [x[0] for x in avg_ratings.columns]
    st.bar_chart(data = avg_ratings, x_label='Courses', y_label='Average Rating', color=colors[len(avg_ratings)])
st.write('\n')
button2 = st.button('Click to refresh data')

if button2: 
    rated_courses = rated_courses.iloc[0:0]
    st.dataframe(rated_courses, hide_index=True)
st.write('\n\n')   
button3 = st.button('Click to delete your rating')
if button3: 
    rated_courses = rated_courses.iloc[1:]
    st.dataframe(rated_courses, hide_index=True)
    avg_ratings = rated_courses.groupby('Course').agg(['mean'])
    avg_ratings.columns = [x[0] for x in avg_ratings.columns]
    st.bar_chart(data = avg_ratings, x_label='Courses', y_label='Average Rating', color=colors[len(avg_ratings)])
rated_courses.to_csv('ratedCourses.csv')
