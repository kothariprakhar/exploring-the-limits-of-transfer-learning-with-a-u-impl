!pip install transformers datasets torch scikit-learn matplotlib seaborn accelerate -q

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import time

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
class Config:
    MODEL_NAME = "t5-small"
    BATCH_SIZE = 16
    EPOCHS = 2
    MAX_LEN_INPUT = 128
    MAX_LEN_TARGET = 5 # "positive" or "negative" are short
    LEARNING_RATE = 2e-4
    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(Config.SEED)
print(f"Using device: {Config.DEVICE}")

# ==========================================
# 2. DATASET PREPARATION (The Unified Text-to-Text Logic)
# ==========================================
# We use SST-2 (Stanford Sentiment Treebank) via Hugging Face.
# Core Paper Logic: We convert a Classification Task (0/1) into a Text Generation Task.
# Input: "sst2 sentence: <sentence>" -> Target: "positive" or "negative"

def load_and_process_data():
    print("Loading SST-2 dataset...")
    dataset = load_dataset("glue", "sst2")
    
    # Map integer labels to text strings
    label_map = {0: "negative", 1: "positive"}
    
    tokenizer = T5Tokenizer.from_pretrained(Config.MODEL_NAME, legacy=False)

    def preprocess_function(examples):
        # T5 specific prefix for task specification
        inputs = [f"sst2 sentence: {sentence}" for sentence in examples["sentence"]]
        targets = [label_map[label] for label in examples["label"]]
        
        # Tokenize inputs
        model_inputs = tokenizer(inputs, max_length=Config.MAX_LEN_INPUT, padding="max_length", truncation=True, return_tensors="pt")
        
        # Tokenize targets
        labels = tokenizer(targets, max_length=Config.MAX_LEN_TARGET, padding="max_length", truncation=True, return_tensors="pt")
        
        # Replace padding token id's of the labels by -100 so it's ignored by the loss
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        
        model_inputs["labels"] = labels
        return model_inputs

    # Apply processing
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["sentence", "label", "idx"])
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    
    return tokenized_datasets, tokenizer

tokenized_datasets, tokenizer = load_and_process_data()

train_loader = DataLoader(tokenized_datasets["train"].shuffle(seed=Config.SEED).select(range(2000)), batch_size=Config.BATCH_SIZE)
val_loader = DataLoader(tokenized_datasets["validation"], batch_size=Config.BATCH_SIZE)

# ==========================================
# 3. MODEL INITIALIZATION
# ==========================================
# T5 is an Encoder-Decoder Transformer.
model = T5ForConditionalGeneration.from_pretrained(Config.MODEL_NAME)
model.to(Config.DEVICE)

optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
total_steps = len(train_loader) * Config.EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# ==========================================
# 4. TRAINING LOOP
# ==========================================
print("Starting Training...")
train_losses = []

for epoch in range(Config.EPOCHS):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(Config.DEVICE)
        attention_mask = batch["attention_mask"].to(Config.DEVICE)
        labels = batch["labels"].to(Config.DEVICE)
        
        optimizer.zero_grad()
        
        # Forward pass: T5ForConditionalGeneration handles the shifting of labels internally
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        if step % 50 == 0:
            print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f}")
            
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}. Time: {time.time() - start_time:.2f}s")

# ==========================================
# 5. EVALUATION & VISUALIZATION
# ==========================================
print("Starting Evaluation...")
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(Config.DEVICE)
        attention_mask = batch["attention_mask"].to(Config.DEVICE)
        # Labels are currently token IDs with -100, we need the raw text for comparison or clean token ids
        # Since we just want to generate, we ignore passed labels for generation
        
        generated_ids = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            max_length=Config.MAX_LEN_TARGET
        )
        
        # Decode generated outputs
        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        predictions.extend(preds)
        
        # Decode actual labels (handling the -100 replacement)
        lbls = batch["labels"].cpu().numpy()
        lbls = np.where(lbls != -100, lbls, tokenizer.pad_token_id)
        refs = tokenizer.batch_decode(lbls, skip_special_tokens=True)
        actuals.extend(refs)

# Post-processing: Clean up text (T5 might generate extra spaces)
predictions = [p.strip() for p in predictions]
actuals = [a.strip() for a in actuals]

# Metrics
acc = accuracy_score(actuals, predictions)
print(f"Validation Accuracy: {acc:.4f}")

# --- PLOT 1: Training Loss ---
plt.figure(figsize=(10, 5))
plt.plot(range(1, Config.EPOCHS + 1), train_losses, marker='o', label='Training Loss')
plt.title("T5 Fine-Tuning Loss on SST-2")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()

# --- PLOT 2: Confusion Matrix ---
# We only expect "positive" and "negative". If the model hallucinates, it appears as 'Other'.
unique_labels = sorted(list(set(actuals + predictions)))
cm = confusion_matrix(actuals, predictions, labels=unique_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
plt.title("Confusion Matrix: Text-to-Text Classification")
plt.xlabel("Predicted Text")
plt.ylabel("Actual Text")
plt.show()

# --- Display Samples ---
df_results = pd.DataFrame({"Actual": actuals, "Predicted": predictions})
print("Sample Predictions:")
print(df_results.head(10))