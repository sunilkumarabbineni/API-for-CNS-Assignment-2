import logging
from flask import Flask, request, jsonify
from huggingface_hub import InferenceClient
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
import requests
from io import BytesIO
import os
import time
import psutil
import torch
import json

# Setup logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)

# Initialize Hugging Face Inference API Client with your API key
hf_client = InferenceClient(api_key="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")

# Load the pipelines
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
summarizer = pipeline("summarization", model="Abdulkader/autotrain-medical-reports-summarizer-2484176581")
image_classifier = pipeline("image-classification", model="subh71/medical")

# Load fine-tuned disease prediction model
model_path = "disease_symptom_predictor"  # Path to your saved model
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Metrics tracking
request_count = 0
error_count = 0

@app.before_request
def track_request_metrics():
    global request_count
    request_count += 1
    logging.info(f"Total requests so far: {request_count}")
    
    # Log resource usage
    cpu_usage = psutil.cpu_percent()
    memory_usage = psutil.virtual_memory().percent
    logging.info(f"CPU usage: {cpu_usage}% | Memory usage: {memory_usage}%")

def log_error(endpoint):
    global error_count
    error_count += 1
    logging.error(f"Error in {endpoint}. Total errors: {error_count}")

def log_latency(endpoint, start_time):
    latency = time.time() - start_time
    logging.info(f"{endpoint} latency: {latency:.2f} seconds")


# Load label mapping (id2label) for disease predictions
with open("id2label.json", "r") as f:
    id2label = json.load(f)

# New endpoint for disease prediction
@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    start_time = time.time()
    try:
        data = request.json
        symptoms = data.get("symptoms")
        if not symptoms:
            return jsonify({"error": "Symptoms are required"}), 400

        # Tokenize input
        inputs = tokenizer(symptoms, return_tensors="pt")

        # Model prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_label = outputs.logits.argmax(dim=-1).item()
            predicted_disease = id2label[str(predicted_label)]

        # Log latency and return the prediction
        log_latency("/predict_disease", start_time)
        return jsonify({"disease": predicted_disease})

    except Exception as e:
        log_error("/predict_disease")
        return jsonify({"error": str(e)}), 500

# Define the endpoint for summarizing medical documents
@app.route('/summarize', methods=['POST'])
def summarize():
    start_time = time.time()
    try:
        document = request.json.get('document')
        if not document:
            return jsonify({"error": "No document provided"}), 400
        summary = summarizer(document, max_length=200, min_length=100, do_sample=False)
        log_latency("/summarize", start_time)
        return jsonify({"summary": summary[0]['summary_text']})
    except Exception as e:
        log_error("/summarize")
        return jsonify({"error": str(e)}), 500

# Define the endpoint for question answering
@app.route('/ask_question', methods=['POST'])
def ask_question():
    start_time = time.time()
    try:
        question = request.json.get('question')
        if not question:
            return jsonify({"error": "Question not provided"}), 400
        context = load_context_from_file('context.txt')
        answer = qa_pipeline(question=question, context=context)
        log_latency("/ask_question", start_time)
        return jsonify({"answer": answer['answer']})
    except Exception as e:
        log_error("/ask_question")
        return jsonify({"error": str(e)}), 500

# Define the endpoint for medical image classification
@app.route('/classify_image', methods=['POST'])
def classify_image():
    start_time = time.time()
    try:
        image_url = request.json.get('image_url')
        if not image_url:
            return jsonify({"error": "No image URL provided"}), 400
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch image from URL"}), 400
        image = Image.open(BytesIO(response.content))
        predictions = image_classifier(image)
        log_latency("/classify_image", start_time)
        return jsonify({"predictions": predictions})
    except Exception as e:
        log_error("/classify_image")
        return jsonify({"error": str(e)}), 500

# Define the endpoint for extracting entities
@app.route('/extract_entities', methods=['POST'])
def extract_entities():
    start_time = time.time()
    try:
        text = request.json.get('text')
        if not text:
            return jsonify({"error": "No text provided"}), 400
        entities = ner_pipeline(text)
        entities_serializable = [{**entity, 'score': float(entity['score'])} for entity in entities]
        log_latency("/extract_entities", start_time)
        return jsonify({"entities": entities_serializable})
    except Exception as e:
        log_error("/extract_entities")
        return jsonify({"error": str(e)}), 500

# Define the endpoint for sentiment analysis
@app.route('/sentiment', methods=['POST'])
def sentiment_analysis():
    start_time = time.time()
    try:
        feedback = request.json.get('feedback')
        if not feedback:
            return jsonify({"error": "No feedback provided"}), 400
        sentiment = sentiment_analyzer(feedback)
        sentiment_label = 'positive' if sentiment[0]['label'] == 'POSITIVE' else 'negative'
        log_latency("/sentiment", start_time)
        return jsonify({"sentiment": sentiment_label})
    except Exception as e:
        log_error("/sentiment")
        return jsonify({"error": str(e)}), 500

# Define the endpoint for the chatbot
@app.route('/chat', methods=['POST'])
def chat():
    start_time = time.time()
    try:
        user_input = request.json.get('user_input')
        if not user_input:
            return jsonify({"error": "No user input provided"}), 400
        messages = [{"role": "user", "content": user_input}]
        output = hf_client.chat.completions.create(
            model="microsoft/Phi-3.5-mini-instruct",  
            messages=messages,
            stream=True,
            temperature=0.5,
            max_tokens=1024,
            top_p=0.7
        )
        full_response = []
        for chunk in output:
            full_response.append(chunk.choices[0].delta.content)
        log_latency("/chat", start_time)
        return jsonify({"response": "".join(full_response)})
    except Exception as e:
        log_error("/chat")
        return jsonify({"error": str(e)}), 500

# Utility function to load context from file for QA
def load_context_from_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Run the Flask app locally
if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5004)
