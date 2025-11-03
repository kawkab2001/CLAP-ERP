import torch
from unsloth import FastLanguageModel  # Import from unsloth
from datasets import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import csv
import ast
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.meteor_score import meteor_score
from scipy.stats import rankdata
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType

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
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2-0.5B",  # Use the Qwen2-0.5B model from unsloth
    max_seq_length=128,  # Set a fixed maximum length
    dtype=torch.bfloat16,  # Use bfloat16 for faster computation
    load_in_4bit=True,  # Load the model in 4-bit precision for efficiency
)

def preprocess_data(example):
    try:
        # Add T5-specific prefix for masked language modeling
        masked_text = f"fill-mask: {example['masked_text'].replace('<MASK>', '<extra_id_0>')}"
        # Tokenize the masked text
        encoding = tokenizer(
            masked_text,
            max_length=128,  # Set a fixed maximum length
            padding="max_length",  # Pad to max_length
            truncation=True,  # Truncate if longer than max_length
            return_tensors="pt",  # Return PyTorch tensors
        )
        # Get the positions of the <extra_id_0> tokens
        mask_token_id = tokenizer.additional_special_tokens_ids[0]  # <extra_id_0>
        mask_positions = [i for i, token_id in enumerate(encoding["input_ids"][0]) if token_id == mask_token_id]
        # Tokenize the masked words
        masked_word_ids = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word)) for word in
                           example["masked_words"]]
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
tokenized_dataset = dataset.map(preprocess_data, remove_columns=["original_text", "masked_text", "masked_words"],
                                batched=False)

# Filter out examples where preprocessing returned None
tokenized_dataset = tokenized_dataset.filter(lambda x: x is not None)

# Step 3: Apply PEFT (LoRA) for efficient fine-tuning
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Task type for causal language models
    inference_mode=False,
    r=8,  # Rank of the low-rank matrices
    lora_alpha=32,  # Scaling factor
    lora_dropout=0.1  # Dropout rate
)
model = get_peft_model(model, peft_config)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 4: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./qwen2-finetuned",
    evaluation_strategy="steps",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=12,
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
model.save_pretrained("./qwen2-finetuned")
tokenizer.save_pretrained("./qwen2-finetuned")

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
    plt.savefig('qwen2_train_loss.png')  # Save the plot
    plt.show()

# Extract logs from the trainer
training_logs = trainer.state.log_history

# Plot the losses
plot_loss(training_logs)

# Step 9: Test the Fine-Tuned Model
def test_model(model, tokenizer, masked_sentence):
    """
    Test the fine-tuned Qwen2 model on a masked sentence with multiple <MASK> tokens.
    Args:
        model: The fine-tuned Qwen2 model.
        tokenizer: The tokenizer used for tokenization.
        masked_sentence: A string containing one or more <MASK> tokens.
    Returns:
        A list of predicted words for each masked position.
    """
    # Replace each <MASK> with a unique <extra_id_X> token
    mask_count = masked_sentence.count('<MASK>')
    extra_ids = [f"<extra_id_{i}>" for i in range(mask_count)]
    masked_sentence_with_extra_ids = masked_sentence
    for i, extra_id in enumerate(extra_ids):
        masked_sentence_with_extra_ids = masked_sentence_with_extra_ids.replace('<MASK>', extra_id, 1)
    # Add the T5-specific prefix
    masked_sentence_with_extra_ids = f"fill-mask: {masked_sentence_with_extra_ids}"
    # Tokenize the input
    inputs = tokenizer(masked_sentence_with_extra_ids, return_tensors="pt", padding=True, truncation=True)
    # Move inputs to the same device as the model
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    # Perform inference using the generate method
    with torch.no_grad():
        outputs = model.generate(**inputs)
    # Decode the output to get the predicted words
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
    # Extract predictions corresponding to each <extra_id_X>
    predicted_words = []
    for extra_id in extra_ids:
        start_idx = decoded_output.find(extra_id)
        if start_idx != -1:
            start_idx += len(extra_id)  # Move past the <extra_id_X> token
            end_idx = decoded_output.find('<', start_idx)  # Find the next special token
            if end_idx == -1:
                end_idx = len(decoded_output)  # If no next special token, take the rest of the string
            predicted_word = decoded_output[start_idx:end_idx].strip()
            predicted_words.append(predicted_word)
        else:
            predicted_words.append(None)  # If <extra_id_X> not found, append None
    return predicted_words

# Example usage
test_sentence = "<MASK> accounting <MASK> record <MASK> report process <MASK> closely integrated with almost every other process."
predicted_words = test_model(model, tokenizer, test_sentence)
print(f"Test Sentence: {test_sentence}")
print(f"Predicted Words: {predicted_words}")

# Step 10: Evaluate Model with Metrics
from nltk.tokenize import word_tokenize

def calculate_mrr(predicted_words, actual_words):
    """
    Calculate Mean Reciprocal Rank (MRR).
    """
    for rank, word in enumerate(predicted_words, start=1):
        if word in actual_words:
            return 1 / rank
    return 0

from nltk.translate.bleu_score import sentence_bleu

def evaluate_metrics(model, tokenizer, dataset):
    """
    Evaluate the model using METEOR, MRR, Cosine Similarity, Perplexity, and BLEU.
    """
    example = dataset[:20]
    meteor_scores = []
    mrr_scores = []
    similarities = []
    perplexities = []
    bleu_scores = []  # List to store BLEU scores
    for i in range(len(example)):
        masked_sentence = example["masked_text"][i]
        actual_words = example["masked_words"][i]
        masked_sentence2 = f"fill-mask: {masked_sentence.replace('<MASK>', '<extra_id_0>')}"
        # Tokenize the input
        # Decode the predicted token IDs back to words
        predicted_words = test_model(model, tokenizer, masked_sentence)
        # Ensure predicted_words is a flat list of tokens
        if not isinstance(predicted_words, list):
            predicted_words = [predicted_words]  # Wrap in list if necessary
        # Replace None with "MASK" in predicted_words
        predicted_words = ["MASK" if item is None else item for item in predicted_words]
        # Tokenize the sentences (though they are already tokenized, this step ensures compatibility)
        hypothesis_str = " ".join(predicted_words)
        reference_str = " ".join(actual_words)
        # Compute METEOR Score
        hypothesis_tokens = word_tokenize(hypothesis_str)
        reference_tokens = [word_tokenize(reference_str)]  # METEOR expects a list of references
        meteor_score_value = meteor_score(reference_tokens, hypothesis_tokens)
        meteor_scores.append(meteor_score_value)
        # Compute MRR
        mrr_score_value = calculate_mrr(predicted_words, actual_words)
        mrr_scores.append(mrr_score_value)
        # Compute Cosine Similarity
        true_embeddings = model.model.embed_tokens.weight[
            [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))[0] for word in actual_words]
        ].detach().cpu().numpy()
        predicted_embeddings = model.model.embed_tokens.weight[
            [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))[0] for word in predicted_words]
        ].detach().cpu().numpy()
        similarity = cosine_similarity(true_embeddings, predicted_embeddings).diagonal().mean()
        similarities.append(similarity)
        # Compute BLEU Score
        reference_tokens_bleu = [word_tokenize(reference_str)]  # BLEU expects a list of references
        hypothesis_tokens_bleu = word_tokenize(hypothesis_str)
        # Debug prints
        print(f"Reference Tokens: {reference_tokens_bleu}")
        print(f"Hypothesis Tokens: {hypothesis_tokens_bleu}")
        # Compute BLEU Score
        bleu_score_value = sentence_bleu(reference_tokens_bleu, hypothesis_tokens_bleu)
        bleu_scores.append(bleu_score_value)
    # Print all metrics
    print(f"Average METEOR Score: {np.mean(meteor_scores):.4f}")
    print(f"Average MRR: {np.mean(mrr_scores):.4f}")
    print(f"Average Cosine Similarity: {np.mean(similarities):.4f}")
    print(f"Average BLEU Score: {np.mean(bleu_scores):.4f}")

# Example usage
evaluate_metrics(model, tokenizer, df)

# Step 11: Test the Model
test_sentence = "The implementation is very <MASK>"
predicted_words = test_model(model, tokenizer, test_sentence)
print(f"Test Sentence: {test_sentence}")
print(f"Predicted Words: {predicted_words}")