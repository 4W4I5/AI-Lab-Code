import os
import librosa
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from multiprocessing import Pool

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
        self.model.cuda()  # Move model to GPU
        self.scaler = StandardScaler()
        self.optimizer = optim.Adam(self.model.parameters())
        self.loss_fn = nn.MSELoss()
    
    def _extract_features_from_directory(self, directory):
        print("Extracting features from directory:", directory)
        features_list = []
        files = os.listdir(directory)
        total_files = sum(1 for filename in files if filename.endswith('.mp3'))
        completed_files = 0
        try:
            with Pool() as p:
                for features in p.imap_unordered(self._process_file, [os.path.join(directory, filename) for filename in files if filename.endswith('.mp3')]):
                    if features is not None:  # Check for None values
                        features_list.append(features)
                    completed_files += 1
                    print(f"Processed {completed_files}/{total_files} files", end='\r')
            print("\nFeature extraction completed.")
        except KeyboardInterrupt:
            print("\nCtrl+C detected. Feature extraction interrupted.")
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
    
    def _process_file(self, audio_file):
        features = {}
        y, sr = librosa.load(audio_file)
        features['Pitch'] = np.mean(librosa.yin(y, fmin=50, fmax=2000))  # Adjust fmin and fmax as needed
        features['Formants'] = np.mean(librosa.effects.harmonic(y))
        features['Intensity'] = np.mean(librosa.feature.rms(y=y))
        features['Duration'] = librosa.get_duration(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        if np.isnan(np.sum(mfcc)):  # Check for NaN values in MFCC
            return None
        features['Spectral Features'] = np.mean(mfcc, axis=1)
        return features
    
    def train(self, train_directory, train_metadata):
        print("Training model...")
        train_features = self._extract_features_from_directory(train_directory)
        train_features.fillna(train_features.mean(), inplace=True)
        
        train_metadata['age'] = train_metadata['age'].apply(self._age_to_numeric)
        train_labels = train_metadata['age'].values
        
        train_features_scaled = torch.tensor(self.scaler.fit_transform(train_features.values), dtype=torch.float32, device='cuda')
        train_labels = torch.tensor(train_labels, dtype=torch.float32, device='cuda')
        
        train_dataset = torch.utils.data.TensorDataset(train_features_scaled, train_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        for epoch in range(10):  # Adjust number of epochs as needed
            self.model.train()
            for i, (features, labels) in enumerate(train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(features)
                loss = self.loss_fn(outputs, labels.view(-1, 1))
                loss.backward()
                self.optimizer.step()
                if (i + 1) % 10 == 0:
                    print(f"Epoch [{epoch+1}/{10}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        print("Training completed.")
    
    def predict(self, test_directory):
        print("Predicting speaker ages...")
        test_features = self._extract_features_from_directory(test_directory)
        test_features.fillna(test_features.mean(), inplace=True)
        
        test_features_scaled = torch.tensor(self.scaler.transform(test_features.values), dtype=torch.float32, device='cuda')
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(test_features_scaled)
        return predictions.cpu().numpy().flatten()
    
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
