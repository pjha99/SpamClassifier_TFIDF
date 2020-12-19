# Project Name
Spam Classification Using NLP 


#### -- Project Status: [Active, On-Hold, Completed]

## Project Intro/Objective
In this Project, we will build and train a model Using NLP techniques and apply Machine Learning 
So that Model is able to predict/classify the SPAM texts.

### Methods Used
* Data Preprocessing
* Used TF-IDF to convert word to vectors.
* Used Naive Bayes as classification Algorithm.

### Technologies
*The Model is developed using Python.
*Libraries Involved are:
1.)Pandas
2.)nltk
3.)sklearn

## Project Description
The example is based on a dataset that is publicly available from the UCI Machine Learning Repository.


Field BareNuc is a categorical feature, we will be using one hot encoding to convert it into the numerical Value.

## Train and Test Data
 80% data is used to train while 205 is used for testing the model.

## Getting Started

1. Raw Data is being kept under the project repo with the name SMSSpamCollection.csv    
2. Code is written under the spamclassifier.py


# RESULTS

#####confusion Matrix#######
[ [955   0]
  [31 129]]

#####classification report#####
      precision    recall  f1-score   support

   0       0.97      1.00      0.98       955
   1       1.00      0.81      0.89       160

    accuracy                           0.97      1115
   macro avg       0.98      0.90      0.94      1115
weighted avg       0.97      0.97      0.97      1115
 
 
#####accuracy######
1.0