#
# Simple solution to first (ungraded) assignment, created with ChatGPT in < 20 minutes
#

import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import nltk
import re
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier # added for experiment 2
import pandas as pd # added for experiment 1.4

# Download necessary resources from nltk
nltk.download('punkt')
nltk.download('stopwords')

data_directory = "/Users/valgardg/Desktop/spamClassification/ling-spam/train-mails"
test_directory = "/Users/valgardg/Desktop/spamClassification/ling-spam/test-mails"

### hr: Interesting, I asked for
###     "dropping all non words like punctuation characters, numbers, and single letters)"
def preprocess_text(text):
    # print("preprocess_text: ", text)
    text = text.replace("Subject:", "") # Remove "Subject:" from the subject line
    text = text.lower() # Convert the text to lower case
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove all non-alphabetic characters
    words = word_tokenize(text) # Tokenize
    words = [word for word in words if len(word) > 1 and word not in stopwords.words('english')] # Remove all single-character words and stopwords
    stemmer = PorterStemmer() # Stemming
    words = [stemmer.stem(word) for word in words] # More Stemming
    # print("processed_text: ", ' '.join(words))
    return ' '.join(words)  # Convert the list of words to a string


def load_spam_data(data_directory):
    data = []
    labels = []

    for filename in os.listdir(data_directory):
        if filename.startswith("spm"):
            labels.append(1)
        else:
            labels.append(0)

        with open(os.path.join(data_directory, filename), 'r', encoding='utf-8') as file:
            email_text = file.read()
            preprocessed_text = preprocess_text(email_text)
            data.append(preprocessed_text)

    return data, labels

def run_experiment(vectorizer, train_data, train_labels, test_data, test_labels):
    # Transform text data to numerical vectors
    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)
    
    print("Experiment 1.4: Logistic Regression with varying C (Train vs Test)")
    
    # Create an empty DataFrame to store results
    results_df = pd.DataFrame(columns=['C', 'Train_Recall', 'Train_Precision', 'Test_Recall', 'Test_Precision'])
    
    # Possible C values for the experiment
    C_values = [0.01, 0.1, 1, 10, 100]
    
    for C in C_values:
        # Initialize Logistic Regression classifier with C
        lr_classifier = LogisticRegression(C=C)
        
        # Train the classifier
        lr_classifier.fit(X_train, train_labels)
        
        # Evaluate metrics on training set
        train_predictions = lr_classifier.predict(X_train)
        train_recall = recall_score(train_labels, train_predictions)
        train_precision = precision_score(train_labels, train_predictions)
        
        # Evaluate metrics on test set
        test_predictions = lr_classifier.predict(X_test)
        test_recall = recall_score(test_labels, test_predictions)
        test_precision = precision_score(test_labels, test_predictions)
        
        print(f"For C = {C}:")
        print(f"  Train Recall: {train_recall}, Train Precision: {train_precision}")
        print(f"  Test Recall: {test_recall}, Test Precision: {test_precision}")

        # Append to results DataFrame
        new_row = pd.DataFrame({'C': [C], 
                        'Train_Recall': [train_recall], 
                        'Train_Precision': [train_precision],
                        'Test_Recall': [test_recall], 
                        'Test_Precision': [test_precision]})
        results_df = pd.concat([results_df, new_row]).reset_index(drop=True)
        
    # Print or save the DataFrame
    print(results_df)

# Load the training data and labels
train_data, train_labels = load_spam_data(data_directory)

# Load the test data and labels
test_data, test_labels = load_spam_data(test_directory)

# Create the CountVectorizer
vectorizer = CountVectorizer(max_features=2000)  # chosen from experiment 1.1

# Run the experiment
run_experiment(vectorizer, train_data, train_labels, test_data, test_labels)

# # Create a CountVectorizer with a maximum vocabulary size of 2000 words
# vectorizer = CountVectorizer(max_features=2000)
# X_train = vectorizer.fit_transform(train_data)
# X_test = vectorizer.transform(test_data)

# # Train a Multinomial Naive Bayes classifier
# classifier = MultinomialNB()
# classifier.fit(X_train, train_labels)

# # Predict on the test data
# test_predictions = classifier.predict(X_test)

# # Evaluate the classifier
# accuracy = accuracy_score(test_labels, test_predictions)
# precision = precision_score(test_labels, test_predictions)
# recall = recall_score(test_labels, test_predictions)
# f1 = f1_score(test_labels, test_predictions)

# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1-score:", f1)

# # Calculate ROC curve
# fpr, tpr, thresholds = roc_curve(test_labels, classifier.predict_proba(X_test)[:, 1])
# roc_auc = auc(fpr, tpr)
# print("AUC-ROC:", roc_auc)

# # Plot ROC curve
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()
