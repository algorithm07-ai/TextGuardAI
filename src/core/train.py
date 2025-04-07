import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
import os
from data_processor import DataProcessor

class SimpleClassifier(nn.Module):
    def __init__(self, input_size=768, hidden_size=256, num_classes=2):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def evaluate_model(model, val_loader, device):
    model.eval()
    val_preds = []
    val_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(features)
            preds = torch.argmax(outputs, dim=1)
            
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    val_preds = np.array(val_preds)
    val_labels = np.array(val_labels)
    
    # Calculate metrics
    report = classification_report(val_labels, val_preds, target_names=['Ham', 'Spam'])
    accuracy = (val_preds == val_labels).mean()
    
    return accuracy, report

def train_model(model, train_loader, val_loader, device, num_epochs=3, learning_rate=1e-3):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    best_val_accuracy = 0
    best_model_path = 'best_model.pt'
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch in progress_bar:
            features = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
        
        # Evaluate on validation set
        val_accuracy, val_report = evaluate_model(model, val_loader, device)
        print(f'\nValidation Accuracy: {val_accuracy:.4f}')
        print('Validation Report:')
        print(val_report)
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f'New best model saved with validation accuracy: {val_accuracy:.4f}')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Load and prepare data
    print('Loading data...')
    df = processor.load_data('SMSSpamCollection')
    train_df, val_df, test_df = processor.prepare_data(df)
    train_loader, val_loader, test_loader = processor.create_dataloaders(train_df, val_df, test_df)
    
    # Initialize model
    print('Initializing model...')
    model = SimpleClassifier().to(device)
    
    # Train model
    print('Starting training...')
    train_model(model, train_loader, val_loader, device)
    
    # Load best model and evaluate on test set
    print('\nEvaluating on test set...')
    model.load_state_dict(torch.load('best_model.pt'))
    test_accuracy, test_report = evaluate_model(model, test_loader, device)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print('Test Report:')
    print(test_report)

if __name__ == '__main__':
    main() 