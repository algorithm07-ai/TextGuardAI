import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset, DataLoader
import torch

class SMSDataset(Dataset):
    def __init__(self, texts, labels, vectorizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.vectorizer = vectorizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Convert text to TF-IDF features
        features = self.vectorizer.transform([text]).toarray()[0]
        features = np.pad(features, (0, max(0, self.max_length - len(features))))

        return {
            'input_ids': torch.FloatTensor(features),
            'labels': label
        }

class DataProcessor:
    def __init__(self, batch_size=16):
        self.batch_size = batch_size
        self.vectorizer = TfidfVectorizer(max_features=768)

    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def load_data(self, file_path):
        # Read the dataset
        df = pd.read_csv(file_path, sep='\t', names=['label', 'message'])
        # Map labels
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        # Preprocess messages
        df['message'] = df['message'].apply(self.preprocess_text)
        return df

    def prepare_data(self, df, test_size=0.2, val_size=0.1):
        # First split into train and test
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        # Then split train into train and validation
        train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=42)
        
        # Fit vectorizer on training data
        self.vectorizer.fit(train_df['message'])
        
        return train_df, val_df, test_df

    def create_dataloaders(self, train_df, val_df, test_df):
        # Create datasets
        train_dataset = SMSDataset(
            texts=train_df['message'].values,
            labels=train_df['label'].values,
            vectorizer=self.vectorizer
        )
        val_dataset = SMSDataset(
            texts=val_df['message'].values,
            labels=val_df['label'].values,
            vectorizer=self.vectorizer
        )
        test_dataset = SMSDataset(
            texts=test_df['message'].values,
            labels=test_df['label'].values,
            vectorizer=self.vectorizer
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

        return train_loader, val_loader, test_loader 