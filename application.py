from transformers import pipeline

nlg_pipeline = pipeline("text-generation", model="distilgpt2")
response = nlg_pipeline("What is the weather like today?")[0]['generated_text']
print(response)
