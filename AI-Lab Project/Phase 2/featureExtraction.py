import os
import random
import time
import warnings

import audioflux as af
import IPython.display as ipd
import joblib
import librosa
import librosa.display
import noisereduce as nr
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchaudio
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


class LinearRegressions:
    def __init__(
        self,
        learning_rate=0.01,
        n_iterations=1000,
        regularization=None,
        reg_strength=0.1,
    ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.reg_strength = reg_strength
        self.weights = None
        self.bias = None

    def _normalize_features(self, X):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_normalized = (X - mean) / std
        return X_normalized

    def _compute_cost(self, X, y):
        n_samples = len(y)
        y_pred = np.dot(X, self.weights) + self.bias
        mse = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)

        if self.regularization == "l2":
            reg_term = (self.reg_strength / (2 * n_samples)) * np.sum(self.weights**2)
            mse += reg_term
        elif self.regularization == "l1":
            reg_term = (self.reg_strength / n_samples) * np.sum(np.abs(self.weights))
            mse += reg_term

        return mse

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Normalize features
        X_normalized = self._normalize_features(X)

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iterations):
            # Predictions
            y_pred = np.dot(X_normalized, self.weights) + self.bias

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X_normalized.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        X_normalized = self._normalize_features(X)
        return np.dot(X_normalized, self.weights) + self.bias


def accuracy(y_true, y_pred):
    """
    Calculate the R-squared accuracy score for regression.

    Parameters:
    - y_true: array-like, true target values
    - y_pred: array-like, predicted target values

    Returns:
    - accuracy: float, R-squared score
    """
    return r2_score(y_true, y_pred)


def preprocess_audio(fname):
    y, sr = librosa.load(path=fname, sr=16_000, mono=True)

    # Preprocessing i.e. Cut silence at start/end, remove bg noise.
    y_trim, _ = librosa.effects.trim(y=y, top_db=10)
    noise_reduced = nr.reduce_noise(y=y_trim, sr=sr)

    # Onset detection
    onsets = librosa.onset.onset_detect(y=noise_reduced, sr=sr, hop_length=128)
    numberOfWords = len(onsets)

    # Length of audio file
    duration = len(y_trim) / sr

    # Fundamental frequency extraction
    f0, _, _ = librosa.pyin(y=y_trim, sr=sr, fmin=10, fmax=8000, frame_length=1024)
    f0_values = [
        np.nanmean(f0),
        np.nanmedian(f0),
        np.nanstd(f0),
        np.nanpercentile(f0, 5),
        np.nanpercentile(f0, 95),
    ]

    # Directly extract the pitch via mean of frequencies
    # f0 = librosa.piptrack(y=y_trim, sr=sr, fmin=10, fmax=8000, threshold=0.75)

    # Additional features
    hnr = librosa.effects.harmonic(y=y_trim)
    spectral_centroid_features = librosa.feature.spectral_centroid(y=y_trim, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_trim, sr=sr)[0]
    spectral_contrast = librosa.feature.spectral_contrast(y=y_trim, sr=sr)[0]

    # Mel spectrogram
    # mel_spectrogram = librosa.feature.melspectrogram(y=y_trim, sr=sr)
    # mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    # mel_strength = np.mean(mel_spectrogram, axis=1)
    # melSpec_length = int(duration * sr / 128)
    # mel_spectrogram = resize_spec(mel_spectrogram, melSpec_length, fact =-80)

    # Send an update of the file being completed
    print(f"\r[Parallelized]: File {fname[-17:]} completed")

    # Update metadata
    return {
        "number_of_words": numberOfWords,
        "duration": duration,
        "words_per_second": numberOfWords / duration,
        "pitch": f0_values[0],
        # "pitch": f0[0],
        # "formant1": spectral_centroid_features[0],
        # "formant2": spectral_centroid_features[1],
        "f0_median": f0_values[1],
        "f0_std": f0_values[2],
        "f0_5th_percentile": f0_values[3],
        "f0_95th_percentile": f0_values[4],
        "hnr": np.mean(hnr),
        "spectral_centroid": np.mean(spectral_centroid_features),
        "spectral_bandwidth": np.mean(spectral_bandwidth),
        "spectral_contrast": np.mean(spectral_contrast),
    }


def load_data(preProcess=False, limitFiles=None):
    train_metadata = pd.read_csv("./Dataset/truncated_train.csv")
    test_metadata = pd.read_csv("./Dataset/cv-valid-test.csv")
    print(f"[Main] Train metadata shape: {train_metadata.shape}")
    print(f"[Main] Test metadata shape: {test_metadata.shape}")
    print(f"[Main] Training Metadata: \n {train_metadata.head()}")
    print(f"[Main] Testing Metadata: \n {test_metadata.head()}")

    # Drop duration column as it is filled later on
    train_metadata.drop(columns=["duration"], inplace=True)
    test_metadata.drop(columns=["duration"], inplace=True)

    # remove rows that have a NaN value in ages column in training data
    train_metadata = train_metadata[train_metadata["age"].notna()].reset_index(
        drop=True
    )
    test_metadata = test_metadata[test_metadata["age"].notna()].reset_index(drop=True)

    print(
        f"\n[Main] Train metadata shape after removing NaN values: {train_metadata.shape}"
    )
    print(
        f"[Main] Train metadata after removing NaN values: \n {train_metadata.head()}"
    )
    age_mapping = {
        "teens": 0,
        "twenties": 1,
        "thirties": 2,
        "fourties": 3,
        "fifties": 4,
        "sixties": 5,
        "seventies": 6,
        "eighties": 7,
        "nineties": 8,
    }
    train_metadata["age"] = train_metadata["age"].map(age_mapping)
    test_metadata["age"] = test_metadata["age"].map(age_mapping)

    # Remap genders, 0 is male 1 is female
    gender_mapping = {"male": 0, "female": 1}
    train_metadata["gender"] = train_metadata["gender"].map(gender_mapping)
    test_metadata["gender"] = test_metadata["gender"].map(gender_mapping)

    # Rename age to age_range
    train_metadata.rename(columns={"age": "age_range"}, inplace=True)
    test_metadata.rename(columns={"age": "age_range"}, inplace=True)
    print(f"\n[Main] Train metadata after remapping: \n {train_metadata.head()}")
    print(f"\n[Main] Test metadata after remapping: \n {test_metadata.head()}")

    if preProcess == False:
        # Preprocess the audio files
        print("[Main] Preprocessing Training audio files...")
        prefix = "./Dataset"

        # ====================== PREPROCSS TRAINING DATA ======================
        # num_files = train_metadata.shape[0]
        if limitFiles:
            num_files = limitFiles
        else:
            num_files = train_metadata.shape[0]

        # Parallelize preprocessing using joblib
        processed_data = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(preprocess_audio)(
                f"{prefix}/{train_metadata['filename'][i]}"
            )
            for i in range(num_files)
        )
        # Update gender based on pitch, if pitch is greater than 165 then it is female, set it to 1, otherwise 0
        for i, data in enumerate(processed_data):
            if data["pitch"] > 165:
                processed_data[i]["gender"] = 1
            else:
                processed_data[i]["gender"] = 0

        # Normalize any NaN values in processed data with mean of the column
        for i, data in enumerate(processed_data):
            for key, value in data.items():
                print(f"In column {key}")
                if np.isnan(value):
                    processed_data[i][key] = train_metadata[key].mean()

        # Update metadata with preprocessed features
        for i, data in enumerate(processed_data):
            for key, value in data.items():
                train_metadata.loc[i, key] = value

        # If limitFiles then shorten DF
        if limitFiles:
            train_metadata[:limitFiles]

        # Write preprocessed metadata to file
        train_metadata.to_csv(
            "./Dataset/preprocessed/trainingPreprocessed.csv", index=False
        )

        # ====================== PREPROCSS TESTING DATA ======================
        if limitFiles:
            num_files1 = limitFiles
        else:
            num_files1 = test_metadata.shape[0]

        print("\n[Main] Preprocessing Testing audio files...")
        # Parallize preprocessing using joblib for testing
        processed_data_test = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(preprocess_audio)(f"{prefix}/{test_metadata['filename'][i]}")
            for i in range(num_files1)
        )
        # Update gender based on pitch, if pitch is greater than 165 then it is female, set it to 1, otherwise 0
        for i, data in enumerate(processed_data_test):
            if data["pitch"] > 165:
                processed_data_test[i]["gender"] = 1
            else:
                processed_data_test[i]["gender"] = 0
        # Normalize any NaN values in processed data with mean of the column
        for i, data in enumerate(processed_data_test):
            for key, value in data.items():
                if np.isnan(value):
                    processed_data_test[i][key] = test_metadata[key].mean()

        # Update metadata with preprocessed features
        for i, data in enumerate(processed_data_test):
            for key, value in data.items():
                test_metadata.loc[i, key] = value

        # Add duration column from processed_data to test metadata
        for i, data in enumerate(processed_data_test):
            test_metadata.loc[i, "duration"] = data["duration"]

        # If limitFiles then shorten the df by limitFiles
        if limitFiles:
            test_metadata[:limitFiles]

        test_metadata.to_csv(
            "./Dataset/preprocessed/testingPreprocessed.csv", index=False
        )

    else:
        # Load preprocessed dataframes
        train_metadata = pd.read_csv("./Dataset/preprocessed/trainingPreprocessed.csv")
        test_metadata = pd.read_csv("./Dataset/preprocessed/testingPreprocessed.csv")

    return train_metadata, test_metadata


def splitData(train_metadata, test_metadata):
    # Select target
    target = "age_range"
    y_train = train_metadata[target].values
    y_test = test_metadata[target].values

    # Drop filename and text columns
    train_metadata.drop(columns=["filename", "text", "accent"], inplace=True)
    test_metadata.drop(columns=["filename", "text", "accent"], inplace=True)

    sns.heatmap(train_metadata.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    sns.heatmap(test_metadata.corr(), annot=True, fmt=".2f", cmap="coolwarm")

    # Select relevant features from the dataframe, ensure filename is dropped
    selected_features = [
        "gender",
        "number_of_words",
        "words_per_second",
        "pitch",
        "f0_median",
        "f0_std",
        "f0_5th_percentile",
        "f0_95th_percentile",
        "hnr",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_contrast",
    ]

    # Training set, only include rows that do not have a Nan value in any column
    train_metadata = train_metadata.dropna()
    test_metadata = test_metadata.dropna()

    x_train = train_metadata[selected_features]
    y_train = train_metadata[target]

    # Testing set
    x_test = test_metadata[selected_features]
    y_test = test_metadata[target]

    print(f"[Main] Training set size: {x_train.shape}")
    print(f"[Main] Testing set size: {x_test.shape}")

    return x_train, y_train, x_test, y_test


def customLinearRegression(x_train, y_train, x_test, y_test):
    # Initialize Linear Regression model
    model= LinearRegressions(
        learning_rate=0.01, n_iterations=1000, regularization="l2", reg_strength=0.1
    )

    # Train the model
    model.fit(x_train, y_train)

    # Predict on the training and testing data
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    # Calculate mean squared error
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print(f"\n[Linear Regression Model]")
    print(f"\tTrain MSE (Mean-Squared Err): {train_mse}")
    print(f"\tTest MSE (Mean-Squared Err): {test_mse}")
    print(f"\tTrain MAE (Mean-Absolute Err): {np.mean(np.abs(y_train - y_train_pred))}")
    print(f"\tTest MAE (Mean-Absolute Err): {np.mean(np.abs(y_test - y_test_pred))}")
    print(f"\tTrain R2 Score: {r2_score(y_train, y_train_pred)}")
    print(f"\tTest R2 Score: {r2_score(y_test, y_test_pred)}")

    # print(f"\n[Linear Regression Model] Results:")
    # if model.score(x_test, y_test) > 0.5:
    #     print(f"\tModel is good")
    # elif model.score(x_test, y_test) > 0.7:
    #     print(f"\tModel is excellent")
    # elif model.score(x_test, y_test) < 0:
    #     print(f"\tModel is bad, possible overfitting")
    # print(f"]tAccuracy Score: {accuracy_score(y_test, y_test_pred)}")

    return model


def main():
    time_start = time.time()

    # Check if the dataset is already preprocessed
    if not os.path.exists("./Dataset/preprocessed"):
        os.makedirs("./Dataset/preprocessed")
        # print(f"set preprocessed to False in first")
        preprocessed = False
    else:
        # Check if the preprocessed files are present
        if not os.path.exists(
            "./Dataset/preprocessed/trainingPreprocessed.csv"
        ) or not os.path.exists("./Dataset/preprocessed/testingPreprocessed.csv"):
            # print(f"set preprocessed to False in second")
            preprocessed = False
        else:
            # print(f"set preprocessed to True in second")
            preprocessed = True

    train_metadata, test_metadata = load_data(preProcess=preprocessed, limitFiles=50)
    print(f"[Main] Training data: \n {train_metadata.head()}")
    print(f"[Main] Testing data: \n {test_metadata.head()}")
    x_train, y_train, x_test, y_test = splitData(train_metadata, test_metadata)

    LinearRegressionModel = customLinearRegression(x_train, y_train, x_test, y_test)

    time_end = time.time()

    print(
        f"\n[Main] Program Execution Complete\n\tTime taken: {time_end - time_start:.2f} seconds."
    )


if __name__ == "__main__":
    main()
