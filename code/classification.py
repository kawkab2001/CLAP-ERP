import random
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
import torch
from transformers import BertTokenizer, BertModel
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# File path and dataset loading
file_path = 'dataset2.txt'

with open(file_path, "r", encoding="utf-8") as file:
    sentences = file.readlines()

# Function to simulate missing words
def simulate_missing_words(sentence, missing_prob=0.2):
    words = nltk.word_tokenize(sentence)
    missing_word_indices = [i for i, word in enumerate(words) if random.random() < missing_prob]
    new_sentence = [word for i, word in enumerate(words) if i not in missing_word_indices]
    return ' '.join(new_sentence), missing_word_indices

# Create dataset with simulated missing words
data = []
for sentence in sentences:
    for _ in range(10):  # Augmenting the dataset
        missing_sentence, missing_indices = simulate_missing_words(sentence)
        is_missing = 1 if len(missing_sentence.split()) < len(sentence.split()) else 0
        data.append({"sentence": missing_sentence, "missing": is_missing, "missing_indices": missing_indices})

# BERT Tokenizer and Model initialization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_bert_embeddings(sentence):
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Extract embeddings and labels
X = [get_bert_embeddings(d['sentence']) for d in data]
y = [d['missing'] for d in data]

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Hyperparameters for models
rf_params = {
    'n_estimators': 100,  # Number of trees
    'max_depth': 10,      # Maximum depth of trees
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42
}

xgb_params = {
    'n_estimators': 100,  # Number of trees
    'max_depth': 6,       # Maximum depth of trees
    'learning_rate': 0.1,  # Learning rate
    'subsample': 0.8,     # Subsample ratio of the training data
    'colsample_bytree': 0.8,  # Fraction of features to consider at each split
    'random_state': 42
}

lgb_params = {
    'n_estimators': 100,  # Number of trees
    'max_depth': 6,       # Maximum depth of trees
    'learning_rate': 0.1,  # Learning rate
    'num_leaves': 31,     # Number of leaves in one tree
    'random_state': 42
}

# Initialize models with hyperparameters
rf_model = RandomForestClassifier(**rf_params)
xgb_model = xgb.XGBClassifier(**xgb_params)
lgb_model = lgb.LGBMClassifier(**lgb_params)

# Function to simulate training over epochs
def train_model_for_epochs(model, X_train, y_train, epochs=5):
    print(f"Training {model.__class__.__name__} model")
    for epoch in range(epochs):
        # Simulate training by fitting the model in each "epoch"
        model.fit(X_train, y_train)
        print(f"Epoch {epoch+1}/{epochs} completed.")

# Train the models over multiple epochs (simulated)
train_model_for_epochs(rf_model, X_train, y_train, epochs=5)
train_model_for_epochs(xgb_model, X_train, y_train, epochs=5)
train_model_for_epochs(lgb_model, X_train, y_train, epochs=5)

# Predict using the models
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)
lgb_pred = lgb_model.predict(X_test)

# Calculate metrics for all models
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    return accuracy, precision, recall, f1, roc_auc

rf_metrics = calculate_metrics(y_test, rf_pred)
xgb_metrics = calculate_metrics(y_test, xgb_pred)
lgb_metrics = calculate_metrics(y_test, lgb_pred)

# Prepare a dataframe for comparison
metrics_df = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'LightGBM'],
    'Accuracy': [rf_metrics[0], xgb_metrics[0], lgb_metrics[0]],
    'Precision': [rf_metrics[1], xgb_metrics[1], lgb_metrics[1]],
    'Recall': [rf_metrics[2], xgb_metrics[2], lgb_metrics[2]],
    'F1 Score': [rf_metrics[3], xgb_metrics[3], lgb_metrics[3]],
    'AUC': [rf_metrics[4], xgb_metrics[4], lgb_metrics[4]],
})

# Plotting metrics comparison
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot for accuracy, precision, recall, F1 score
metrics_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1 Score']].plot(kind='bar', ax=axes[0, 0], color=['skyblue', 'lightgreen', 'salmon', 'lightcoral'])
axes[0, 0].set_title('Comparison of Accuracy, Precision, Recall, F1 Score')

# Plot for AUC
metrics_df.set_index('Model')['AUC'].plot(kind='bar', ax=axes[0, 1], color='lightblue')
axes[0, 1].set_title('AUC Comparison')

# Confusion Matrix Plots
models = [rf_model, xgb_model, lgb_model]
model_names = ['Random Forest', 'XGBoost', 'LightGBM']

for i, model in enumerate(models):
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Missing Words', 'Missing Words'], yticklabels=['No Missing Words', 'Missing Words'], ax=axes[1, i % 2])
    axes[1, i % 2].set_title(f'Confusion Matrix - {model_names[i]}')

# Adjust layout and show
plt.tight_layout()
plt.show()

# Save metrics to CSV
metrics_df.to_csv('plots/model_comparison_metrics.csv', index=False)

# ROC Curve and AUC comparison
plt.figure(figsize=(8, 6))
for model, name in zip([rf_model, xgb_model, lgb_model], ['Random Forest', 'XGBoost', 'LightGBM']):
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('plots/roc_comparison.png')  # Save ROC curve comparison plot
plt.show()
