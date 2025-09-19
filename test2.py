from transformers import ViTForImageClassification, ViTImageProcessor, TrainingArguments, Trainer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset from folder
print("Loading dataset...")
dataset = load_dataset("imagefolder", data_dir="dataset")

# Debug: check dataset features
print("Dataset features:", dataset["train"].features)
print(f"Number of classes: {len(dataset['train'].features['label'].names)}")
print(f"Class names: {dataset['train'].features['label'].names}")
print(f"Train samples: {len(dataset['train'])}")
if "validation" in dataset:
    print(f"Validation samples: {len(dataset['validation'])}")
if "test" in dataset:
    print(f"Test samples: {len(dataset['test'])}")

# Load processor & model
model_name = "google/vit-base-patch16-224"
print(f"Loading model and processor: {model_name}")

processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=len(dataset["train"].features["label"].names),
    ignore_mismatched_sizes=True,
    id2label={str(i): label for i, label in enumerate(dataset["train"].features["label"].names)},
    label2id={label: str(i) for i, label in enumerate(dataset["train"].features["label"].names)}
)

# Move model to device
model.to(device)

# Preprocess function
def transform(example_batch):
    # Process images
    inputs = processor(
        images=[img.convert('RGB') for img in example_batch["image"]], 
        return_tensors="pt"
    )
    # Add labels
    inputs["labels"] = torch.tensor(example_batch["label"], dtype=torch.long)
    return inputs

# Apply transformation
print("Applying transformations...")
prepared_ds = dataset.with_transform(transform)

# Define compute metrics function for evaluation
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Training arguments with corrected parameter names
training_args = TrainingArguments(
    output_dir="./civic_model",
    eval_strategy="epoch",  # Changed from 'evaluation_strategy'
    save_strategy="epoch",
    logging_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    warmup_steps=100,
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,  # Keep only 2 best checkpoints
    seed=42,
    data_seed=42,
    remove_unused_columns=False,  # Important for image processing
    dataloader_pin_memory=False,  # Set to False to avoid potential issues
)

# Create trainer
print("Setting up trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=prepared_ds["train"],
    eval_dataset=prepared_ds["validation"] if "validation" in dataset else None,
    compute_metrics=compute_metrics,
    tokenizer=processor,  # Pass processor as tokenizer for saving
)

# Train the model
print("Starting training...")
try:
    trainer.train()
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed with error: {e}")
    raise e

# Save the final model
print("Saving model...")
trainer.save_model("./civic_model_final")
processor.save_pretrained("./civic_model_final")

# Evaluate on test set if available
if "test" in dataset:
    print("Evaluating on test set...")
    test_results = trainer.evaluate(eval_dataset=prepared_ds["test"])
    print(f"Test results: {test_results}")

print("Training and evaluation complete!")

# Optional: Test inference on a single image
def test_inference():
    """Test inference on a single image from the validation set"""
    if "validation" in dataset:
        # Get a sample image
        sample = dataset["validation"][0]
        image = sample["image"]
        true_label = sample["label"]
        
        # Prepare image
        inputs = processor(images=image.convert('RGB'), return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_id = predictions.argmax().item()
        
        # Get class names
        predicted_class = dataset["train"].features["label"].names[predicted_class_id]
        true_class = dataset["train"].features["label"].names[true_label]
        confidence = predictions[0][predicted_class_id].item()
        
        print(f"\nSample Prediction:")
        print(f"True label: {true_class}")
        print(f"Predicted label: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")

# Run inference test
test_inference()