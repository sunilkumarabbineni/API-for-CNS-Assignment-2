from datasets import load_dataset
import json

# Load the dataset
ds = load_dataset("shanover/disease_symptoms_prec_full")

# Convert the dataset to JSONL format
with open('disease_symptoms_dataset.jsonl', 'w') as f:
    for record in ds['train']:
        json_record = {
            "symptoms": record['symptoms'],
            "disease": record['disease'],
            "precautions" : record['precautions']
        }
        f.write(json.dumps(json_record) + '\n')

print("Dataset has been converted to JSONL format and saved as 'disease_symptoms.jsonl'.")
