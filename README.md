# Sentiment Analysis Project using NLP, Naive Bayes and Logistic Regression
## Table of Contents
1. Project Overview
2. Technologies and Libraries Used
3. Project Workflow
   1. Dataset
   2. Data Preprocessing
   3. Text Vectorization
   4. Class Balancing using SMOTE
   5. Model Building and Training
   6. Evaluation
4. Usage Instructions
   1. Installation
   2. Running the Project
5. Results
6. Conclusion


## 1. Project Overview
This project aims to build a Sentiment Analysis system that can classify text reviews or statements as either positive or negative. The project leverages Natural Language Processing (NLP) techniques and libraries such as NLTK for text preprocessing, followed by applying machine learning models including Naive Bayes and Logistic Regression to classify the sentiments.

We preprocess the text data using standard NLP techniques, convert the text into a numerical format using TF-IDF vectorization, handle class imbalances using SMOTE (Synthetic Minority Oversampling Technique), and train both a Logistic Regression and Naive Bayes classifier to predict sentiment.

## 2. Technologies and Libraries Used
   - Python 3.7+
   - Natural Language Toolkit (NLTK): For text    preprocessing.
   - Scikit-learn: For machine learning algorithms (Naive Bayes, Logistic Regression), train-test split, and evaluation metrics.
   - Imbalanced-learn (imblearn): For handling class imbalance using SMOTE.
   - Pandas: For data manipulation.
   - Numpy: For numerical operations.
   - Matplotlib/Seaborn: For visualizations.
   - Jupyter Notebook: For running the project interactively.
## 3. Project Workflow
### 1. Dataset
The dataset used for this project contains textual reviews along with sentiment labels (either positive or negative). The dataset is has the following columns:
- Review: The raw text review.
- Sentiment: The sentiment label ( positive and negative).
### 2. Data Preprocessing
We utilize NLP and NLTK for text preprocessing:

- Lowercasing: Convert all text to lowercase to avoid case sensitivity.
- Removing Special Characters: Punctuation and special characters are removed.
- Stopwords Removal: Common words with little meaning (e.g., 'the', 'is', etc.) are removed using the NLTK stopwords corpus.
- Lemmatization: Words are reduced to their base forms (e.g., 'running' becomes 'run').

### 3. Text Vectorization
We use TF-IDF (Term Frequency - Inverse Document Frequency) vectorization to convert the cleaned text into numerical features for the machine learning models.

### 4. Class Balancing using SMOTE
To handle class imbalance in the dataset (where one class might be significantly underrepresented), we apply SMOTE to oversample the minority class.

### 5. Model Building and Training
Logistic Regression Model
The Logistic Regression model is trained on the TF-IDF vectorized text data to predict the sentiment of the reviews.

### 6. Evaluation
We evaluate both the models using a classification report and confusion matrix to measure accuracy, precision, recall, and F1-score.

## 4. Usage Instructions

### 1. Installation
- Clone this repository or download the project files.
- Install the required dependencies listed in the requirements.txt file:
   ```
   pip install -r requirements.text
   ```
- If you're using Jupyter Notebook, simply open the notebook and run the cells sequentially.

### 2. Running the Project
- Ensure the dataset is correctly loaded in the script or notebook.
- Preprocess the text using the provided preprocessing function.
- Apply TF-IDF vectorization.
- Handle class imbalance using SMOTE.
- Train both Logistic Regression and Naive Bayes models.
- Evaluate the performance using the classification report and confusion matrix.

## 5. Results
- Logistic Regression: This model tends to perform well on binary classification tasks and often serves as a reliable baseline for sentiment analysis.
- Naive Bayes: A probabilistic classifier that can handle text classification efficiently, though sometimes less accurate than Logistic Regression in certain cases.
- After training and testing both models, you can compare their performance in terms of precision, recall, F1-score, and accuracy using the provided evaluation metrics.

## 6. Conclusion
This project demonstrates how to implement a Sentiment Analysis system using NLP techniques, with two machine learning models (Logistic Regression and Naive Bayes).
