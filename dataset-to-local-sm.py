from datasets import load_dataset
import json

# Load the dataset
ds = load_dataset("shanover/disease_symptoms_prec_full")

# Select the first 1000 records
first_1000_records = ds['train'].select(range(1000))

# Convert the dataset to JSONL format
with open('disease_symptoms_dataset_sm.jsonl', 'w') as f:
    for record in first_1000_records:
        json_record = {
            "symptoms": record['symptoms'],
            "disease": record['disease'],
            "precautions": record['precautions']
        }
        f.write(json.dumps(json_record) + '\n')

print("The first 1000 rows from the dataset have been converted to JSONL format and saved as 'disease_symptoms_dataset_sm.jsonl'.")
