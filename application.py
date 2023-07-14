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
    def process_conversation_file(self, file_path):
        conversation = self.read_conversation(file_path)
        messages = self.parse_conversation(conversation)
        return [self.extract_sender_and_message(message) for message in messages]

    @staticmethod
    def read_conversation(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    @staticmethod
    def parse_conversation(conversation):
        messages = conversation.split('Sender:')
        return [message.strip() for message in messages if message.strip()]

    @staticmethod
    def extract_sender_and_message(message):
        lines = message.split('\n')
        sender = lines[0].strip()
        message = ' '.join(line.strip() for line in lines[2:-1])
        return {'sender': sender, 'message': message}


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
file_name = 'conversations_training_data/raw_conversations/Atos-Paula_Calandrelli'
sender_message_pairs = conversations_files_processor.process_conversation_file(file_name)
spacy_adapter = SpacyAdapter()
features = {'is_remote_job': spacy_adapter.is_remote_job(sender_message_pairs)}
salary_calculator = SalaryCalculator()
salary = salary_calculator.calculate_salary(features)

print(f"Is Remote Job: {features['is_remote_job']}")
print(f"Salary: {salary}")

