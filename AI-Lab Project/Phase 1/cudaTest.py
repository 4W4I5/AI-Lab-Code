import os
import librosa
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class CustomLinearRegression(nn.Module):
    def __init__(self, input_size):
        super(CustomLinearRegression, self).__init__()
        self.fc = nn.Linear(input_size, 1)
    
    def forward(self, x):
        x = self.fc(x)
        return x

class SpeakerAgePredictor:
    def __init__(self, input_size):
        self.model = CustomLinearRegression(input_size)
        self.scaler = StandardScaler()
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = nn.MSELoss()
    
    def _extract_features_from_directory(self, directory):
        features_list = []
        for filename in os.listdir(directory):
            if filename.endswith('.mp3'):
                audio_file = os.path.join(directory, filename)
                features = self._extract_features(audio_file)
                features_list.append(features)
        return pd.DataFrame(features_list)
    
    def _age_to_numeric(self, age_str):
        age_mapping = {
            'teens': 1,
            'twenties': 2,
            'thirties': 3,
            'fourties': 4,
            'fifties': 5,
            'sixties': 6,
            'seventies': 7,
            'eighties': 8,
            'nineties': 9
        }
        return age_mapping[age_str]
    
    def _extract_features(self, audio_file):
        y, sr = librosa.load(audio_file)
        pitch = np.mean(librosa.yin(y, fmin=50, fmax=2000))  # Adjust fmin and fmax as needed
        formants = np.mean(librosa.effects.harmonic(y))
        intensity = np.mean(librosa.feature.rms(y=y))
        duration = librosa.get_duration(y=y, sr=sr)
        spectral_features = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        return {
            'Pitch': pitch,
            'Formants': formants,
            'Intensity': intensity,
            'Duration': duration,
            'Spectral Features': spectral_features
        }
    
    def train(self, train_directory, train_metadata):
        train_features = self._extract_features_from_directory(train_directory)
        train_features.fillna(train_features.mean(), inplace=True)
        
        train_metadata['age'] = train_metadata['age'].apply(self._age_to_numeric)
        train_labels = train_metadata['age'].values
        
        train_features_scaled = torch.tensor(self.scaler.fit_transform(train_features.values), dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.float32)
        
        train_dataset = torch.utils.data.TensorDataset(train_features_scaled, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        for epoch in range(10):  # Adjust number of epochs as needed
            self.model.train()
            for features, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.loss_fn(outputs, labels.view(-1, 1))
                loss.backward()
                self.optimizer.step()
    
    def predict(self, test_directory):
        test_features = self._extract_features_from_directory(test_directory)
        test_features.fillna(test_features.mean(), inplace=True)
        
        test_features_scaled = torch.tensor(self.scaler.transform(test_features.values), dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(test_features_scaled)
        return predictions.numpy().flatten()
    
    def evaluate(self, test_labels, predictions):
        mse = mean_squared_error(test_labels, predictions)
        mae = mean_absolute_error(test_labels, predictions)
        r2 = r2_score(test_labels, predictions)
        return mse, mae, r2

# Usage example
if __name__ == "__main__":
    input_size = 17  # Adjust according to the number of features extracted
    predictor = SpeakerAgePredictor(input_size)
    
    # Replace placeholders with actual paths and labels
    train_directory = "./Dataset/cv-valid-train"
    test_directory = "./Dataset/cv-valid-test"
    train_metadata = pd.read_csv("./Dataset/truncated_train.csv")
    test_metadata = pd.read_csv("./Dataset/cv-valid-test.csv")
    
    predictor.train(train_directory, train_metadata)
    
    # Example: Assuming we want to predict the age of speakers in the test set
    predictions = predictor.predict(test_directory)
    test_labels = test_metadata['age'].apply(predictor._age_to_numeric).values
    
    mse, mae, r2 = predictor.evaluate(test_labels, predictions)
    
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R-squared:", r2)
