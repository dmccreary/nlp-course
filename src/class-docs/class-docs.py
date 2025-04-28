import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Function to read documents from folders
def read_documents(positive_folder, negative_folder):
    documents = []
    labels = []
    
    # Read positive documents
    for filename in os.listdir(positive_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(positive_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(content)
                labels.append(1)  # Positive class
    
    # Read negative documents
    for filename in os.listdir(negative_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(negative_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(content)
                labels.append(0)  # Negative class
    
    return documents, labels

# Main function
def main():
    # Define paths to document folders
    positive_folder = "./positive_documents"
    negative_folder = "./negative_documents"
    
    # Read documents
    print("Reading documents...")
    documents, labels = read_documents(positive_folder, negative_folder)
    
    # Create a dataframe
    df = pd.DataFrame({
        'text': documents,
        'label': labels
    })
    
    # Print dataset statistics
    print(f"Total documents: {len(df)}")
    print(f"Positive documents: {sum(df['label'] == 1)}")
    print(f"Negative documents: {sum(df['label'] == 0)}")
    
    # Split the dataset into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    print(f"Training set size: {len(train_df)}")
    print(f"Testing set size: {len(test_df)}")

    # Initialize BERT tokenizer and model
    model_name = "roberta-base"  # You can use other models like "bert-base-uncased", "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Tokenize the text
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    # Prepare datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Tokenize datasets
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Define trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    # Train the model
    print("Training the model...")
    trainer.train()
    
    # Evaluate the model
    print("Evaluating the model...")
    eval_result = trainer.evaluate()
    print(f"Evaluation results: {eval_result}")
    
    # Make predictions on test set
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(test_df['label'], preds))
    
    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(test_df['label'], preds))
    
    # Save the model
    trainer.save_model("./saved_model")
    tokenizer.save_pretrained("./saved_model")
    print("Model saved to ./saved_model")
    
    # Function to predict on new documents
    def predict_document(document_text):
        inputs = tokenizer(document_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        return pred_class, confidence
    
    # Example usage of prediction function
    print("\nExample prediction:")
    example_doc = "This is an example document for prediction."
    pred_class, confidence = predict_document(example_doc)
    print(f"Predicted class: {'Positive' if pred_class == 1 else 'Negative'}")
    print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()