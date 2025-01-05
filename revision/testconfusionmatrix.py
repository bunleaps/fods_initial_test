import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("output_conversations.csv")
df = df.dropna(subset=['text', 'label'])

# Define a function for stratified sampling
def stratified_sample(df, label_column, samples_per_label):
    # Create an empty list to store the sampled data
    sampled_data = []
    
    # Group by the label column
    grouped = df.groupby(label_column)
    
    # Loop through each group and sample
    for _, group in grouped:
        # Sample from each group
        sampled_data.append(group.sample(n=samples_per_label, random_state=42))
    
    # Concatenate all sampled groups into a single DataFrame
    return pd.concat(sampled_data).reset_index(drop=True)

# Set the number of samples per label (modifiable variable)
samples_per_label = 1800

# Sample the data
sampled_df = stratified_sample(df, label_column='label', samples_per_label=samples_per_label)

# Define features (X) and labels (y)
X = sampled_df['text']
y = sampled_df['label']

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=10000)
X_vectorized = vectorizer.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate on the test set
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
labels = sorted(y.unique())  # Get sorted class labels
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

##########################################
#            Model Testing             #
##########################################

# Load the chat data from the JSON file
with open('testing_data.json', 'r') as f:
    chat_data = json.load(f)

# Print header with better alignment
print("Model Testing: ")
print(f"{'prediction':<15}{'expected':<14}")
print("-----------------------")

# Process the messages as before
for chat in chat_data:
    joined_message = " ".join(chat["messages"])
    joined_vectorized = vectorizer.transform([joined_message])

    # Classify the combined test message
    predicted_label = clf.predict(joined_vectorized)

    # Print the results with the match column
    print(f"{predicted_label[0]:>10}{chat['expected_label']:>13}")
