import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

##########################################
#           Data Processing             #
##########################################

# Load dataset
df = pd.read_csv("output_conversations.csv")

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
samples_per_label = 712

# Sample the data
sampled_df = stratified_sample(df, label_column='label', samples_per_label=samples_per_label)

##########################################
#      Random Forest Classifier Setup   #
##########################################

# Define features (X) and labels (y)
X = sampled_df['text']
y = sampled_df['label']

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=10000)
X_vectorized = vectorizer.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
clf = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=50)
clf.fit(X_train, y_train)

##########################################
#           Model Evaluation            #
##########################################
# Define the positive class
positive_class = 'harmful'

# Evaluate on the test set
y_pred = clf.predict(X_test)
# precision = precision_score(y_test, y_pred, pos_label=positive_class)
# recall = recall_score(y_test, y_pred, pos_label=positive_class)
# f1 = f1_score(y_test, y_pred, pos_label=positive_class)
accuracy = accuracy_score(y_test, y_pred)

# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1-Score: {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

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