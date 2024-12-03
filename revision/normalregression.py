import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("output_conversations.csv")
df = df.dropna(subset=['text', 'label'])

# Define features (X) and labels (y)
X = df['text']
y = df['label']

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

# Test on new chat messages
chat_messages = [
    "Hey, I don't like what you're doing.",
    "You're just wasting my time!",
    "I hate people like you."
]
joined_message = " ".join(chat_messages)
joined_vectorized = vectorizer.transform([joined_message])

# Classify the combined test message
predicted_label = clf.predict(joined_vectorized)
print("\nPredicted Label for Combined Messages:", predicted_label[0], "harmful")

chat_messages = [
    "Hey, I really appreciate your effort!",
    "You're doing a great job, keep it up!",
    "I admire people like you."
]

joined_message = " ".join(chat_messages)
joined_vectorized = vectorizer.transform([joined_message])

# Classify the combined test message
predicted_label = clf.predict(joined_vectorized)
print("\nPredicted Label for Combined Messages:", predicted_label[0], "safe")

chat_messages = [
    "Hey!", "What's up?", 
    "Heyyy. Nothing much. Kust chillin. U?", 
    "Same. Watchin some tv.", 
    "Nice nice. Whatchu watchin?", 
    "Parks and Rec.", 
    "I love that show!", 
    "Treat yo self!", 
    "Haha yeah it's so good"
]

joined_message = " ".join(chat_messages)
joined_vectorized = vectorizer.transform([joined_message])

# Classify the combined test message
predicted_label = clf.predict(joined_vectorized)
print("\nPredicted Label for Combined Messages:", predicted_label[0], "safe")

chat_messages = [
    "would u like to meet rn?", 
    "to do what", 
    "anything u like", 
    "what would u like to do?", 
    "i dunno", 
    "something not boring", 
    "lol", 
    "do u like to mess around?",
    "lol like how",
    "kissing",
    "touching",
    "heh..yeah..",
    "would u like to meet and mess around?",
    "maybe",
    "lol"
]

joined_message = " ".join(chat_messages)
joined_vectorized = vectorizer.transform([joined_message])

# Classify the combined test message
predicted_label = clf.predict(joined_vectorized)
print("\nPredicted Label for Combined Messages:", predicted_label[0], "harmful")

chat_messages = [
    "i wanna see a picture of you right this second",
    "what kind of pic",
    "?",
    "just a normal picture"
]

joined_message = " ".join(chat_messages)
joined_vectorized = vectorizer.transform([joined_message])

# Classify the combined test message
predicted_label = clf.predict(joined_vectorized)
print("\nPredicted Label for Combined Messages:", predicted_label[0], "harmful")

chat_messages = [
    "Aw thank you so much !!",
    "And just to let you know.. If you know anyone that ever thought about modeling for VS. I am still recruiting for our new amateur modeling program",
    "Models getting paid $300,000 a shoot, no prior modeling experience needed",
    "That's interesting! Can you give me more details on how it works?"
]

joined_message = " ".join(chat_messages)
joined_vectorized = vectorizer.transform([joined_message])

# Classify the combined test message
predicted_label = clf.predict(joined_vectorized)
print("\nPredicted Label for Combined Messages:", predicted_label[0], "harmful")

chat_messages = [
    "Hey! I saw your profile and I think you’d be perfect for a great opportunity we have right now.",
    "Oh, really? What kind of opportunity is it?",
    "We’re currently recruiting people for luxury spas and high-end resorts. The pay is amazing—up to $300,000 a year, and everything’s taken care of for you.",
    "Wow, that sounds like a lot of money. What would I have to do?",
    "It’s really simple. Just work in a beautiful environment, with no experience needed. We’ll provide accommodation, food, and transport. You just need to get started.",
    "That sounds good, but I’m a bit skeptical. Is this legitimate? What’s the catch?",
    "There’s no catch! It’s an incredible offer, but spots are limited. Just pay a small registration fee to get all your paperwork done, and we’ll handle the rest.",
    "Hmm, I don’t know. Why would I need to pay anything upfront? I’m not sure about that.",
    "I understand your hesitation, but this is standard for international programs like this. Once you're in, you’ll be making a lot more money than you paid. It’s a small investment for a big future.",
    "I suppose that makes sense. How soon would I have to leave if I decide to join?",
    "We can have you set up and flying out in just a few days. It’s a fast-track program for those ready to seize this opportunity.",
    "Okay, I’ll think about it and let you know soon."
]

joined_message = " ".join(chat_messages)
joined_vectorized = vectorizer.transform([joined_message])

# Classify the combined test message
predicted_label = clf.predict(joined_vectorized)
print("\nPredicted Label for Combined Messages:", predicted_label[0], "harmful")

chat_messages = [
    "You fine pass Kim Kardashian",
    "I swear",
    "Wey this your cloth sef",
    "Lol",
    "Afi kim Kardashian na",
    "Lool",
    "The cloth dey o",
    "You make sense jo",
    "You no dey show off",
    "You for don get like 1 million followers for intagram",
    "Lol"
]

joined_message = " ".join(chat_messages)
joined_vectorized = vectorizer.transform([joined_message])

# Classify the combined test message
predicted_label = clf.predict(joined_vectorized)
print("\nPredicted Label for Combined Messages:", predicted_label[0], "safe")

chat_messages = [
    "Another daily tip from valerie!",
    "When trying to get around ur grandparents with a tub of ice cream do not stuff it up ur shirt",
    "Ur welcome",
    "Thank the lords I have a friend like you!!!",
    "Ur welcome ill always be here if u need advice",
    "Yep"
]

joined_message = " ".join(chat_messages)
joined_vectorized = vectorizer.transform([joined_message])

# Classify the combined test message
predicted_label = clf.predict(joined_vectorized)
print("\nPredicted Label for Combined Messages:", predicted_label[0], "safe")