import pandas as pd

df = pd.read_csv('output_conversations.csv')
df = df.dropna(subset=['text', 'label'])
label_counts = df['label'].value_counts()
print(label_counts)