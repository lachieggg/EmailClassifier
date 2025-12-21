import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm

class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_emails(data_dir):
    """Load emails from directory structure."""
    texts = []
    labels = []
    
    # Load unwanted emails (label = 1)
    unwanted_dir = os.path.join(data_dir, 'DeleteEmail')
    if os.path.exists(unwanted_dir):
        for filename in os.listdir(unwanted_dir):
            filepath = os.path.join(unwanted_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    texts.append(f.read())
                    labels.append(1)
    
    # Load normal emails (label = 0)
    normal_dir = os.path.join(data_dir, 'KeepEmail')
    if os.path.exists(normal_dir):
        for filename in os.listdir(normal_dir):
            filepath = os.path.join(normal_dir, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    texts.append(f.read())
                    labels.append(0)
    
    return texts, labels

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    return np.array(predictions), np.array(true_labels)

def main():
    # Configuration
    DATA_DIR = './data'
    MODEL_NAME = 'distilbert-base-uncased'
    MAX_LENGTH = 512
    BATCH_SIZE = 8
    EPOCHS = 4
    LEARNING_RATE = 2e-5
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading emails...")
    print(f"Looking in directory: {os.path.abspath(DATA_DIR)}")
    print(f"Checking for:")
    print(f"  - {os.path.abspath(os.path.join(DATA_DIR, 'DeleteEmail'))}")
    print(f"  - {os.path.abspath(os.path.join(DATA_DIR, 'KeepEmail'))}")
    
    texts, labels = load_emails(DATA_DIR)
    
    if len(texts) == 0:
        print("\n❌ ERROR: No emails found!")
        print("\nPlease ensure your directory structure is:")
        print("  ./DeleteEmail/")
        print("    1, 2, 3, ...")
        print("  ./KeepEmail/")
        print("    1, 2, 3, ...")
        print("\nOr update DATA_DIR in the script to point to the correct location.")
        return
    
    print(f"Loaded {len(texts)} emails")
    print(f"  - Unwanted emails: {sum(labels)}")
    print(f"  - Normal emails: {len(labels) - sum(labels)}")
    
    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"\nTrain set: {len(train_texts)} emails")
    print(f"Test set: {len(test_texts)} emails")
    
    # Initialize tokenizer and model
    print("\nInitializing model...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2
    ).to(device)
    
    # Create datasets and dataloaders
    train_dataset = EmailDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    test_dataset = EmailDataset(test_texts, test_labels, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Average training loss: {train_loss:.4f}")
        
        # Evaluate
        predictions, true_labels = evaluate(model, test_loader, device)
        print("\nClassification Report:")
        print(classification_report(
            true_labels, 
            predictions, 
            target_names=['Normal', 'Unwanted']
        ))
        print("\nConfusion Matrix:")
        print(confusion_matrix(true_labels, predictions))
    
    # Save model
    print("\nSaving model...")
    model.save_pretrained('./email_filter_distilbert_model')
    tokenizer.save_pretrained('./email_filter_distilbert_model')
    print("Model saved to ./email_filter_model")

if __name__ == "__main__":
    main()