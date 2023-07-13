import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Define the dataset
data = {'is_remote': [1, 1, 0, 0, 0, 1, 0, 1, 1, 0], 'interested': [1, 1, 0, 0, 0, 1, 0, 1, 1, 0]}
df = pd.DataFrame(data)

# Split the dataset into features and target
x = df['is_remote']
x = x.values.reshape(-1, 1)  # reshape the 1 dimension data to convert it into a column.
y = df['interested']

# Create a decision tree classifier and fit it to the data
clf = DecisionTreeClassifier()
clf.fit(x, y)

# Predict whether a job is of interest or not
job1 = [[1]]  # remote job
job2 = [[0]]  # non-remote job

print(clf.predict(job1))
print(clf.predict(job2))
