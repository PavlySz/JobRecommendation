import pandas as pd 
df = pd.read_csv('jobs_data.csv')
titles = df.title.drop_duplicates().dropna()

print(len(titles))

substring = "data sci"

suggestions = [title for title in titles if title.lower().find(substring.lower()) != -1]
print(suggestions[:5])