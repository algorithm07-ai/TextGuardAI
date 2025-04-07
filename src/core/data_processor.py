import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset, DataLoader
import torch
import logging

logger = logging.getLogger(__name__)

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
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess the input text by removing special characters,
        converting to lowercase, and removing extra whitespace.
        """
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            return text
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            raise
            
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a file and return as a DataFrame.
        """
        try:
            df = pd.read_csv(file_path, sep='\t', names=['label', 'text'])
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """
        Prepare data for training by preprocessing text and fitting the vectorizer.
        """
        try:
            # Preprocess all texts
            df['processed_text'] = df['text'].apply(self.preprocess_text)
            
            # Fit and transform the vectorizer
            X = self.vectorizer.fit_transform(df['processed_text'])
            y = (df['label'] == 'spam').astype(int)
            
            return X, y
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise

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