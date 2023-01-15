from typing import Any, Dict, List, Text
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from transformers import pipeline


class GenerateResponseAction(Action):
    def name(self) -> Text:
        return "action_generate_response"

    def run(self,
            dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        user_input = tracker.latest_message.get("text")
        text_gen_pipeline = pipeline("question-answering")
        all_possible_responses = text_gen_pipeline(user_input, max_length=100, top_p=0.9, top_k=50)
        first_response = all_possible_responses[0]["generated_text"]
        dispatcher.utter_message(first_response)
        return all_possible_responses
