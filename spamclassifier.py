import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
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

# Load the training data and labels
train_data, train_labels = load_spam_data(data_directory)

# Load the test data and labels
test_data, test_labels = load_spam_data(test_directory)

# Create the CountVectorizer
vectorizer = CountVectorizer(max_features=2000)  # chosen from experiment 1.1

def run_comparison(vectorizer, train_data, train_labels, test_data, test_labels):
    # Transform text data to numerical vectors
    X_train = vectorizer.fit_transform(train_data)
    X_test = vectorizer.transform(test_data)
    
    print("Experiment 1.5: Model Comparison")
    
    # Models to compare: Naive Bayes, k-NN, and Logistic Regression
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(C=0.1), # chosen for experiment 1.3
        'k-NN': KNeighborsClassifier(n_neighbors=10), # chosen from experiment 1.2
        'Random Forest': RandomForestClassifier(n_estimators=100)
    }
    
    plt.figure(figsize=(10, 8))

    for name, model in models.items():
        # Train the model
        model.fit(X_train, train_labels)
        
        # Predict on the test set
        test_predictions = model.predict(X_test)
        
        # Evaluate metrics
        accuracy = accuracy_score(test_labels, test_predictions)
        precision = precision_score(test_labels, test_predictions)
        recall = recall_score(test_labels, test_predictions)
        f1 = f1_score(test_labels, test_predictions)
        
        # Calculate and plot ROC curve
        fpr, tpr, thresholds = roc_curve(test_labels, model.predict_proba(X_test)[:, 1])
        roc_auc = auc(fpr, tpr)
        
        print(f"{name}:")
        print(f"  Accuracy: {accuracy}")
        print(f"  Precision: {precision}")
        print(f"  Recall: {recall}")
        print(f"  F1-score: {f1}")
        print(f"  AUC-ROC: {roc_auc}")

        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

run_comparison(vectorizer, train_data, train_labels, test_data, test_labels)