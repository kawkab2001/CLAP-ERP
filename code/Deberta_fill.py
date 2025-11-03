import torch
from transformers import DebertaTokenizer, DebertaForMaskedLM, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import csv
import ast
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.meteor_score import meteor_score
from scipy.stats import rankdata
import numpy as np

# Step 1: Load Data
data = []
with open("masked_dataset.csv", mode='r', encoding='utf-8') as file:
    reader = csv.reader(file)
    headers = next(reader)  # Skip header row
    for row in reader:
        original_text, masked_text, masked_words_str = row
        masked_words = ast.literal_eval(masked_words_str)  # Convert string to list
        if masked_words:  # Only append rows with valid masked_words
            data.append({
                "original_text": original_text,
                "masked_text": masked_text,
                "masked_words": masked_words
            })

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(data)

# Step 2: Tokenization
tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")

def preprocess_data(example):
    try:
        # Replace <MASK> with [MASK] token
        masked_text = example["masked_text"].replace("<MASK>", tokenizer.mask_token)
        
        # Tokenize the masked text
        encoding = tokenizer(
            masked_text,
            max_length=128,  # Set a fixed maximum length
            padding="max_length",  # Pad to max_length
            truncation=True,  # Truncate if longer than max_length
            return_tensors="pt",  # Return PyTorch tensors
        )
        
        # Get the positions of the [MASK] tokens
        mask_token_id = tokenizer.mask_token_id
        mask_positions = [i for i, token_id in enumerate(encoding["input_ids"][0]) if token_id == mask_token_id]
        
        # Tokenize the masked words
        masked_word_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)) for word in example["masked_words"]]
        
        # Debugging: Print details to identify mismatches
        print(f"Masked Text: {masked_text}")
        print(f"Number of [MASK] tokens: {len(mask_positions)}")
        print(f"Number of masked words: {len(masked_word_ids)}")
        
        # Ensure the number of mask positions matches the number of masked words
        if len(mask_positions) != len(masked_word_ids):
            print(f"Skipping example due to mismatch: {len(mask_positions)} masks vs {len(masked_word_ids)} words")
            return None
        
        # Prepare labels (only update the mask positions)
        labels = [-100] * len(encoding["input_ids"][0])  # Initialize labels with -100
        for pos, word_ids in zip(mask_positions, masked_word_ids):
            labels[pos] = word_ids[0]  # Use the first token ID of each masked word
        
        # Add labels to the encoding
        encoding["labels"] = torch.tensor(labels)
        encoding = {key: value.squeeze(0) for key, value in encoding.items()}  # Remove batch dimension
        return encoding
    
    except Exception as e:
        print(f"Error processing example: {e}")
        return None

# Apply preprocessing to the dataset
dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(preprocess_data, remove_columns=["original_text", "masked_text", "masked_words"], batched=False)

# Filter out examples where preprocessing returned None
tokenized_dataset = tokenized_dataset.filter(lambda x: x is not None)

# Step 3: Load the Model
model = DebertaForMaskedLM.from_pretrained("microsoft/deberta-base")

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 4: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./deberta-finetuned",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",  # Directory for storing logs
    logging_steps=10,  # Log every 10 steps
)

# Step 5: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
)

# Step 6: Train the Model
trainer.train()

# Step 7: Save the Model
model.save_pretrained("./deberta-finetuned")
tokenizer.save_pretrained("./deberta-finetuned")

# Step 8: Plot Training and Evaluation Loss
def plot_loss(training_logs):
    # Extract training losses
    train_losses = [log["loss"] for log in training_logs if "loss" in log]
    epochs = range(1, len(train_losses) + 1)  # Use the length of train_losses for epochs

    # Plot the training losses
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss", linestyle="--")  # Remove marker="o"
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    # Save and display the plot
    plt.savefig('deberta_train_loss.png')  # Save the plot
    plt.show()
 
# Extract logs from the trainer
training_logs = trainer.state.log_history

# Plot the losses
plot_loss(training_logs)

# Step 9: Test the Fine-Tuned Model
def test_model(model, tokenizer, masked_sentence):
    """
    Test the fine-tuned model on a masked sentence.
    Args:
        model: The fine-tuned DeBERTa model.
        tokenizer: The tokenizer used for tokenization.
        masked_sentence: A string containing <MASK> tokens.
    Returns:
        Predicted words for the masked positions.
    """
    # Replace <MASK> with [MASK] token
    masked_sentence = masked_sentence.replace("<MASK>", tokenizer.mask_token)
    
    # Tokenize the input
    inputs = tokenizer(masked_sentence, return_tensors="pt", padding=True, truncation=True)
    
    # Move inputs to the same device as the model
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    
    # Get the positions of [MASK] tokens
    mask_token_id = tokenizer.mask_token_id
    mask_positions = [i for i, token_id in enumerate(inputs["input_ids"][0]) if token_id == mask_token_id]
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predictions for the [MASK] positions
    predictions = outputs.logits[0][mask_positions]
    predicted_token_ids = torch.argmax(predictions, dim=-1).tolist()
    
    # Decode the predicted token IDs back to words
    predicted_words = [tokenizer.decode([token_id]) for token_id in predicted_token_ids]
    return predicted_words

# Step 10: Evaluate Model with Metrics
def evaluate_metrics(model, tokenizer, dataset):
    """
    Evaluate the model using METEOR, MRR, Cosine Similarity, and Perplexity.
    """
    meteor_scores = []
    mrr_scores = []
    similarities = []
    perplexities = []
    
    for example in dataset:
        masked_sentence = example["masked_text"]
        actual_words = example["masked_words"]
        print("good")
        # Replace <MASK> with [MASK] token
        masked_sentence = masked_sentence.replace("<MASK>", tokenizer.mask_token)
        print("good")
        # Tokenize the input
        inputs = tokenizer(masked_sentence, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        print("good")
        # Get the positions of [MASK] tokens
        mask_token_id = tokenizer.mask_token_id
        mask_positions = [i for i, token_id in enumerate(inputs["input_ids"][0]) if token_id == mask_token_id]
        print("good")
        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)
        print("good")
        # Get logits for the [MASK] positions
        mask_logits = outputs.logits[0][mask_positions]
        predicted_token_ids = torch.argmax(mask_logits, dim=-1).tolist()
        predicted_words = [tokenizer.decode([token_id]) for token_id in predicted_token_ids]
        print("good")
        # Compute METEOR Score
        meteor_score_value = meteor_score([actual_words], predicted_words)
        meteor_scores.append(meteor_score_value)
        print("good")
        # Compute MRR
        ranks = rankdata([-logit.max().item() for logit in mask_logits], method="min")
        mrr_score_value = np.mean(1 / ranks)
        mrr_scores.append(mrr_score_value)
        print("good")
        # Compute Cosine Similarity
        true_embeddings = model.deberta.embeddings.word_embeddings.weight[
            [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))[0] for word in actual_words]
        ].detach().cpu().numpy()
        predicted_embeddings = model.deberta.embeddings.word_embeddings.weight[
            predicted_token_ids
        ].detach().cpu().numpy()
        similarity = cosine_similarity(true_embeddings, predicted_embeddings).diagonal().mean()
        similarities.append(similarity)
        print("good")
        # Compute Perplexity
        log_probs = torch.nn.functional.log_softmax(mask_logits, dim=-1)
        true_token_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))[0] for word in actual_words]
        true_log_probs = log_probs[range(len(true_token_ids)), true_token_ids].cpu().numpy()
        perplexity = np.exp(-np.mean(true_log_probs))
        perplexities.append(perplexity)
    print("good")
    # Print Metrics
    print(f"Average METEOR Score: {np.mean(meteor_scores):.4f}")
    print(f"Average MRR: {np.mean(mrr_scores):.4f}")
    print(f"Average Cosine Similarity: {np.mean(similarities):.4f}")
    print(f"Average Perplexity: {np.mean(perplexities):.4f}")

# Evaluate Metrics
evaluate_metrics(model, tokenizer, data)
print("good")
# Step 11: Test the Model
test_sentence = "The implementation is very <MASK>"
predicted_words = test_model(model, tokenizer, test_sentence)
print(f"Test Sentence: {test_sentence}")
print(f"Predicted Words: {predicted_words}")