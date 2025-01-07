import pandas as pd
# import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("output_conversations.csv")
df = df.dropna(subset=['text', 'label'])

# Stratified sampling function
def stratified_sample(df, label_column, samples_per_label):
    sampled_data = []
    grouped = df.groupby(label_column)
    for _, group in grouped:
        sampled_data.append(group.sample(n=samples_per_label, random_state=42))
    return pd.concat(sampled_data).reset_index(drop=True)

# Set samples per label
samples_per_label = 1800
sampled_df = stratified_sample(df, label_column='label', samples_per_label=samples_per_label)

# Features and labels
X = sampled_df['text']
y = sampled_df['label']

# Map string labels to numerical values
y = y.map({'safe': 0, 'harmful': 1})

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=10000)
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Define models
models = [
    ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000)),
    ('Decision Tree', DecisionTreeClassifier(random_state=42)),
    ('Extra Trees', ExtraTreesClassifier(random_state=42)),
    ('MLP Classifier', MLPClassifier(random_state=42, max_iter=1000)),
    ('Naive Bayes', MultinomialNB()),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('Stacking Classifier', StackingClassifier(estimators=[
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ], final_estimator=LogisticRegression(), cv=5)),
    ('SVM', SVC(probability=True, random_state=42))
]

# Plot combined ROC AUC curves
plt.figure(figsize=(10, 8))

for name, clf in models:
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)  # Specify pos_label
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Combined ROC AUC Curves')
plt.legend(loc='lower right')
plt.show()
