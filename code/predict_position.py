import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import nltk
import os
from nltk.tokenize import word_tokenize
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2

# Create directory for saving plots
os.makedirs("plots", exist_ok=True)

# Download punkt tokenizer
nltk.download('punkt')

# Load and preprocess the dataset
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()
    return [s.strip() for s in sentences if s.strip()]

def simulate_missing_words(sentences):
    data = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        if len(words) > 2:  # Ensure sentence has enough words
            missing_idx = random.randint(0, len(words) - 1)
            masked_sentence = words[:missing_idx] + ["<MASK>"] + words[missing_idx + 1:]
            data.append((" ".join(masked_sentence), missing_idx))
    return data

def prepare_data(data, tokenizer, max_len):
    sentences, labels = zip(*data)
    sequences = tokenizer.texts_to_sequences(sentences)
    padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')
    return np.array(padded_sequences), np.array(labels)

# Load dataset
file_path = "dataset2.txt"
sentences = load_data(file_path)

data = simulate_missing_words(sentences)

# Tokenization
all_sentences = [s for s, _ in data]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_sentences)
vocab_size = len(tokenizer.word_index) + 1

# Define max length
max_len = max(len(word_tokenize(s)) for s in all_sentences)

# Prepare data
X, y = prepare_data(data, tokenizer, max_len)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

def build_improved_model(vocab_size, max_len):
    model = Sequential([
        Embedding(input_dim=vocab_size,
                  output_dim=128,
                  input_length=max_len,
                  mask_zero=True),

        Bidirectional(LSTM(128,
                           return_sequences=True,
                           kernel_regularizer=l2(0.01),
                           recurrent_dropout=0.2)),
        BatchNormalization(),
        Dropout(0.3),

        Bidirectional(LSTM(64,
                           kernel_regularizer=l2(0.01),
                           recurrent_dropout=0.2)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),

        Dense(1, activation='linear')
    ])

    return model

# Build BiLSTM model with improvements
model = build_improved_model(vocab_size, max_len)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

# Train model with callbacks
history = model.fit(
    X_train, y_train,
    epochs=44,
    batch_size=8,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr]
)

# Predict on test set
y_pred = model.predict(X_test).flatten()

# Compute evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save metrics to CSV
metrics_df = pd.DataFrame({"MAE": [mae], "MSE": [mse], "R-Squared": [r2]})
metrics_df.to_csv("plots/model_metrics.csv", index=False)

# --- Plotting Section ---

# 1. KDE Plot: True vs Predicted Positions
plt.figure(figsize=(10, 6))
sns.kdeplot(y_test, label='True Positions', fill=True)
sns.kdeplot(y_pred, label='Predicted Positions', fill=True)
plt.title('KDE Plot: True vs Predicted Positions')
plt.xlabel('Position Index')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.savefig(r"plots/kde_plot.png")
plt.show()