import os
import pandas
from sklearn.model_selection import train_test_split

# Generate list of conversations
training_files_names = os.listdir('conversations_training_data/preprocessed_conversations')
conversations = []
for file_name in training_files_names:
    with open(f'conversations_training_data/preprocessed_conversations/{file_name}', 'r', encoding='UTF-8') as file:
        conversation = file.read()
        conversations.append(conversation)

# Convert the list of conversations into a pandas DataFrame
conversations_dataframe = pandas.DataFrame({'conversation': conversations})

# Split the DataFrame into training and testing sets
train_data, test_data = train_test_split(conversations_dataframe, test_size=0.2, random_state=42)

# Save the training and testing sets to files
train_data.to_csv('conversations_training_data/train_data.csv', index=False)
test_data.to_csv('conversations_training_data/test_data.csv', index=False)
