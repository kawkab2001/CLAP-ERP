import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import nltk
import scipy.stats as stats
import matplotlib.pyplot as plt
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from tensorflow.keras.layers import GRU
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model with GRU instead of LSTM
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
    Bidirectional(GRU(128, return_sequences=False)),  # Replaced LSTM with GRU
    Dropout(0.3),  # Dropout for regularization
    Dense(64, activation='relu'),  # Dense layer
    Dense(1, activation='linear')  # Predicting position index
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

# Train model with callbacks
history = model.fit(
    X_train, y_train,
    epochs=50,  # Increased epochs for small datasets
    batch_size=16,  # Smaller batch size for better generalization
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
metrics_df.to_csv("plots/model_metrics_gru.csv", index=False)

# === Plotting ===
# 1. Loss & MAE Curves
plt.figure(figsize=(7, 5))

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs. Validation Loss')
plt.legend()
plt.savefig('plots/loss_curve_gru.png')  # Save the first plot

plt.figure(figsize=(7, 5))
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.title('Training vs. Validation MAE')
plt.legend()
plt.savefig('plots/mae_curve_gru.png')  # Save the second plot

# 2. Predicted vs Actual Values
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('Actual Position')
plt.ylabel('Predicted Position')
plt.title('Predicted vs. Actual Values')
plt.axline([0, 0], [1, 1], color='red', linestyle='--')
plt.savefig('plots/predicted_vs_actual_gru.png')  # Save the third plot

# 3. Residual Plot
plt.figure(figsize=(7, 5))
residuals = y_test - y_pred

# Q-Q plot for residuals
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Residuals")
plt.savefig('plots/qq_plot_residuals_gru.png')  # Save the fourth plot

plt.show()