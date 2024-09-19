# SentimentAnalysisProj
This project involves using reviews of courses to generate a numeric rating for the course. In the notebooks, I trained various models using both Paragraph Vector (Doc2Vec) and Bag of words (CountVectorizer) techniques. The most suitable model was implemented in the web app.

## Data Preparation
Data of course reviews and ratings was collected from Stanford Course review catalog, Coursera, Kaggle and Data world. Basic visualization was done to see the best courses according to average rating on coursera. The text data was cleaned and preprocessed into suitable format for encoding.

## Data Encoding
The data was encoded or vectorized using Bag of words (BOW) technique for individual words, bigrams and trigrams. The Paragraph vector was used to create vectors of size 20 from each data instance, treating them as documents. These two encoding formats were used as predictors for models in each case and the evaluation of each format compared.

## Modelling
For effectiveness and simplicity, I trained the three most popular (basic) ML models for NLP: Logistic regression, XGBoost and Naive Bayes. The logistic regression performed slightly better than XGBoost both on BOW and Doc2Vec. The Log reg is also preferred for its quick inference speed and compatibility with probabilistic outputs. 
Calibration was attempted on the model, but only made the model less calibrated.

## Summary
The best model- Log reg on BOW, was used in the streamlit website. Feel free to check out the performance, and comment your suggestions on how to improve the app and more functions to add.
