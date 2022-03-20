import pandas as pd

curated_freq = pd.read_csv('data/curated_freq.csv')
docs = pd.read_csv('data/docs.csv')

docs = docs[docs.err != 1]

print(docs.words_counts.iloc[0][0])