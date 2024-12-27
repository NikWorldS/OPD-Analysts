import google.generativeai as genai
import json
import os

genai.configure(api_key="AI_API_KEY_HERE")

SOURCE_MODEL = "models/gemini-1.5-flash-001-tuning"
TUNED_MODEL_ID = "custom-laptop-advisor-v3"
DISPLAY_NAME = "Custom Laptop Advisor"
DESCRIPTION = "A model fine-tuned for providing PC and laptop recommendations."
TRAINING_DATA_PATH = "training_datas\\first_training_data.jsonl"

if not os.path.exists(TRAINING_DATA_PATH):
    raise FileNotFoundError(f"Training data file not found at {TRAINING_DATA_PATH}")

with open(TRAINING_DATA_PATH, "r", encoding='utf-8') as file:
    training_data = [json.loads(line.strip()) for line in file]

HYPERPARAMETERS = {
    "epoch_count": 5,
    "batch_size": 16,
    "learning_rate": 0.001
}

# Запуск fine-tuning
try:
    operation = genai.create_tuned_model(
        source_model=SOURCE_MODEL,
        training_data=training_data,  # Передача данных напрямую
        id=TUNED_MODEL_ID,
        display_name=DISPLAY_NAME,
        description=DESCRIPTION,
        epoch_count=HYPERPARAMETERS["epoch_count"],
        batch_size=HYPERPARAMETERS["batch_size"],
        learning_rate=HYPERPARAMETERS["learning_rate"],
        input_key="input",  # Входные данные в JSONL
        output_key="output"  # Выходные данные в JSONL
    )
    print(f"Fine-tuning job created with operation ID: {operation.operation.name}")
    print("You can track the progress or wait for completion.")
except Exception as e:
    print(f"An error occurred while creating the fine-tuning job: {e}")
