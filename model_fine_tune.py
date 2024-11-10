from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import json

# 1. Load the Dataset from the JSONL file
dataset = load_dataset('json', data_files='disease_symptoms_dataset.jsonl')
train_dataset = dataset['train']

# Extract unique diseases and map them to integer labels
disease_labels = train_dataset.unique("disease")
label2id = {label: i for i, label in enumerate(disease_labels)}
id2label = {i: label for i, label in enumerate(disease_labels)}

# Save label2id and id2label mappings to JSON
with open("label2id.json", "w") as f:
    json.dump(label2id, f)

with open("id2label.json", "w") as f:
    json.dump(id2label, f)

# Choose a base model (like BERT or DistilBERT)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    # Tokenize the symptoms text and map the disease to an integer label
    inputs = tokenizer(examples["symptoms"], truncation=True, padding="max_length", max_length=128)
    inputs["labels"] = [label2id[d] for d in examples["disease"]]
    return inputs

# Tokenize the dataset
tokenized_dataset = train_dataset.map(tokenize_function, batched=True)

# Initialize the model for classification
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=len(disease_labels),
    id2label=id2label,
    label2id=label2id
)

# Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="no"
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save the Model
trainer.save_model("disease_symptom_predictor")

# Example input for testing
input_text = "itching,skin_rash,nodal_skin_eruptions"
inputs = tokenizer(input_text, return_tensors="pt")

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    predicted_label = outputs.logits.argmax(dim=-1).item()
    predicted_disease = id2label[predicted_label]
    
print("Predicted Disease:", predicted_disease)