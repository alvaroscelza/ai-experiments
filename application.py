import os

import spacy
from spacy.cli import download
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SalaryCalculator:
    base_price = 5000
    three_interviews = 1000
    not_remote_penalty = 1500
    no_AI_project = 800
    timezone_specific = 500
    no_python = 500
    not_owned_product = 500
    have_to_use_clients_computer = 400
    no_english = 300
    each_additional_interview = 200
    uninteresting_project = 200
    cannot_work_on_personal_investigations = 200

    def calculate_salary(self, features):
        salary = self.base_price
        if not features['is_remote_job']:
            salary += self.not_remote_penalty
        return salary


class ConversationFilesProcessor:
    def process_conversation_files(self, folder_path):
        conversation_list = []

        # Iterate over the files in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            # Read the conversation from the file
            conversation = self.read_conversation(file_path)

            # Extract the remote status from the file
            is_remote =self. extract_remote_status(file_path)

            # Create a dictionary for the conversation and remote status
            conversation_dict = {'conversation': conversation, 'is_remote': is_remote}

            # Add the dictionary to the conversation list
            conversation_list.append(conversation_dict)

        return conversation_list

    @staticmethod
    def read_conversation(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    @staticmethod
    def extract_remote_status(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                if line.startswith('remote: '):
                    return line.strip().split(':')[1].strip().lower() == 'true'

        # Default to False if no remote status is found
        return False


class SpacyAdapter:
    model_name = "en_core_web_sm"

    def __init__(self):
        try:
            self.natural_language_processor = spacy.load(self.model_name)
        except OSError:
            download(self.model_name)
            self.natural_language_processor = spacy.load(self.model_name)

    def is_remote_job(self, conversation):
        analyzer = SentimentIntensityAnalyzer()
        remote_keywords = ["remote", "hybrid", "work from home"]
        sentiment_scores = []
        for message in conversation:
            doc = self.natural_language_processor(message["message"])
            for token in doc:
                if token.text.lower() in remote_keywords:
                    sentiment_scores.append(analyzer.polarity_scores(token.text)["compound"])
        overall_sentiment = sum(sentiment_scores)
        return overall_sentiment > 0


conversations_files_processor = ConversationFilesProcessor()
folder_path = 'conversations_training_data/'
conversations = conversations_files_processor.process_conversation_files(folder_path)

