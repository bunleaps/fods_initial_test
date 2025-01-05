import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Vectorize the text data
vectorizer = TfidfVectorizer(max_features=10000)
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train, y_train)

# Predictions and evaluation
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC AUC
y_prob = clf.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
roc_auc = roc_auc_score(y_test, y_prob)
print("\nROC AUC:", roc_auc)

##########################################
#            Model Testing             #
##########################################

# Load chat data
with open('testing_data.json', 'r') as f:
    chat_data = json.load(f)

print("\nModel Testing: ")
print(f"{'prediction':<15}{'expected':<14}")
print("-----------------------")

# Test the model on the new chat data
for chat in chat_data:
    joined_message = " ".join(chat["messages"])
    joined_vectorized = vectorizer.transform([joined_message])
    predicted_label = clf.predict(joined_vectorized)
    print(f"{predicted_label[0]:>10}{chat['expected_label']:>13}")
