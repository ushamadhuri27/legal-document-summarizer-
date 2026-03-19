import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import joblib
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

import pandas as pd
from io import StringIO

# Simulate reading the existing CSV data
csv_data = """caseType,clientType,caseDescription
Consumer Protection,Individual,A consumer alleges that a company sold them a defective product.
Labor,Individual,An employee alleges that they were wrongfully terminated from their job.
... (rest of your data) ...
"""

# Load existing CSV into DataFrame
df = pd.read_csv(StringIO(csv_data))

# Create new Divorce case entries
new_cases = pd.DataFrame([
    {
        "caseType": "Divorce",
        "clientType": "Individual",
        "caseDescription": "I'm filing for divorce because my spouse and I have irreconcilable differences. I'm seeking custody of my children and fair division of assets."
    },
    {
        "caseType": "Divorce",
        "clientType": "Small Business",
        "caseDescription": "I'm a small business owner going through a divorce. I'm concerned about how the division of assets might affect my business operations and ownership."
    },
    {
        "caseType": "Divorce",
        "clientType": "Large Corporation",
        "caseDescription": "One of our executives is going through a high-profile divorce. We are concerned about how the publicity and legal proceedings could affect our company's reputation and internal affairs."
    }
])

# Append new cases
df = pd.concat([df, new_cases], ignore_index=True)

# Advanced text preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Tokenization
        tokens = nltk.word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(tokens)
    return ''

# Load and preprocess data
data = pd.read_csv('ClientClassifyData.csv', encoding='latin1')
print("Columns in the dataset:", data.columns.tolist())  # Debug print

# Prepare features and labels
X = data['caseDescription'].values
y = data['clientType'].values

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model pipeline with hyperparameter tuning
def create_pipeline(model):
    return Pipeline([
        ('vectorizer', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ('model', model)
    ])

# Define models with their parameter grids
models = {
    'naive_bayes': {
        'model': MultinomialNB(),
        'params': {
            'model__alpha': [0.1, 0.5, 1.0]
        }
    },
    'logistic_regression': {
        'model': LogisticRegression(max_iter=1000, n_jobs=-1),
        'params': {
            'model__C': [0.1, 1.0, 10.0],
            'model__class_weight': ['balanced', None]
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(n_jobs=-1),
        'params': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 10, 20],
            'model__min_samples_split': [2, 5]
        }
    }
}

# Find best model using GridSearchCV
best_score = 0
best_model = None
best_model_name = None

for name, model_config in models.items():
    print(f"\nTraining {name}...")
    pipeline = create_pipeline(model_config['model'])
    grid_search = GridSearchCV(
        pipeline,
        model_config['params'],
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Evaluate on test set
    y_pred = grid_search.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} - Test Accuracy: {accuracy:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    if accuracy > best_score:
        best_score = accuracy
        best_model = grid_search.best_estimator_
        best_model_name = name

print(f"\nBest model: {best_model_name} with accuracy: {best_score:.4f}")

# Save the best model and label encoder
joblib.dump(best_model, 'client.joblib')
joblib.dump(label_encoder, 'client_label_encoder.joblib')

# Test with new sentences
new_sentences = [
    "I want to Breakup with my husband",
    "Two brothers were tenant of a landlord in a commercial property.One brother had one son and a daughter (both minor) when he got divorced with his wife.The children's went into mother's custody at the time of divorce and after some years the husband (co tenant) also died. Now can the children of the deceased brother(co tenant) claim the right",
    "Our company is being sued for money laundering using foreign shell companies. The plaintiffs allege that the corporation used a network of shell companies to move billions of dollars in illicit proceeds from its various business operations around the world.The corporation denies all of the allegations. It claims that the shell companies were used for legitimate business purposes, such as protecting its intellectual property and trade secrets. It also claims that the funds were not illicit proceeds, but rather legitimate profits from its business operations",
    "The government alleges that My shop has been stockpiling essential goods, such as food and fuel, in order to create a shortage and drive up prices. My business denies these allegations and claim that it is simply stocking up on inventory in order to meet the needs of its customers. This is not setting prices artificially high, and that its prices are simply reflecting the increased cost of goods."
]

# Preprocess new sentences
new_sentences = [preprocess_text(sentence) for sentence in new_sentences]

# Get predictions with probabilities
predictions = best_model.predict_proba(new_sentences)

print("\nPredictions for new sentences:")
for sentence, prob in zip(new_sentences, predictions):
    top_values = []
    labels = label_encoder.classes_
    for label, probability in zip(labels, prob):
        if len(top_values) < 3:
            top_values.append([round(probability, 4), label])
        else:
            tops = [i[0] for i in top_values]
            min_value = min(tops)
            if probability > min_value:
                min_index = tops.index(min_value)
                top_values[min_index] = [round(probability, 4), label]

    print("----------------------------------------")
    print(sentence)
    print("----------------------------------------")
    for t in sorted(top_values, reverse=True):
        print(f"{t[1]} - {round(100 * t[0], 2)}%")
