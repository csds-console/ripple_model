import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from datasets import load_dataset
import os
from pathlib import Path
import pandas as pd

class CivicModelTester:
    def __init__(self, model_path="./civic_model_final"):
        """
        Initialize the model tester
        
        Args:
            model_path: Path to the saved model directory
        """
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.class_names = ['BrokenStreetLight', 'DrainageOverFlow', 'GarbageNotOverflow', 
                           'GarbageOverflow', 'NoPotHole', 'NotBrokenStreetLight', 'PotHole']
        
        # Load model and processor
        self.load_model()
    
    def load_model(self):
        """Load the trained model and processor"""
        try:
            print(f"Loading model from {self.model_path}...")
            self.model = ViTForImageClassification.from_pretrained(self.model_path)
            self.processor = ViTImageProcessor.from_pretrained(self.model_path)
            self.model.eval()  # Set to evaluation mode
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure the model path is correct and the model has been trained.")
            return False
        return True
    
    def predict_single_image(self, image_path, show_image=True):
        """
        Predict a single image
        
        Args:
            image_path: Path to the image file
            show_image: Whether to display the image
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_class_id = probabilities.argmax().item()
                confidence = probabilities[0][predicted_class_id].item()
            
            predicted_class = self.class_names[predicted_class_id]
            
            # Display results
            print(f"\nPrediction for {image_path}:")
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.4f}")
            
            # Show top 3 predictions
            top_3 = torch.topk(probabilities, 3)
            print("\nTop 3 predictions:")
            for i, (prob, idx) in enumerate(zip(top_3.values[0], top_3.indices[0])):
                print(f"{i+1}. {self.class_names[idx]}: {prob:.4f}")
            
            if show_image:
                plt.figure(figsize=(8, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(image)
                plt.title(f"Input Image")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                probs = probabilities[0].cpu().numpy()
                bars = plt.bar(range(len(self.class_names)), probs)
                plt.xlabel('Classes')
                plt.ylabel('Probability')
                plt.title('Prediction Probabilities')
                plt.xticks(range(len(self.class_names)), self.class_names, rotation=45)
                
                # Highlight the predicted class
                bars[predicted_class_id].set_color('red')
                
                plt.tight_layout()
                plt.show()
            
            return predicted_class, confidence, probabilities[0].cpu().numpy()
            
        except Exception as e:
            print(f"Error predicting image {image_path}: {e}")
            return None, None, None
    
    def test_on_dataset(self, dataset_path="dataset"):
        """
        Test the model on the entire dataset
        
        Args:
            dataset_path: Path to the dataset directory
        """
        print("Loading test dataset...")
        try:
            dataset = load_dataset("imagefolder", data_dir=dataset_path)
            
            # Use test set if available, otherwise use validation set
            if "test" in dataset:
                test_dataset = dataset["test"]
                print(f"Testing on {len(test_dataset)} test samples")
            elif "validation" in dataset:
                test_dataset = dataset["validation"]
                print(f"Testing on {len(test_dataset)} validation samples")
            else:
                print("No test or validation set found!")
                return
            
            predictions = []
            true_labels = []
            
            print("Making predictions...")
            for i, sample in enumerate(test_dataset):
                if i % 100 == 0:
                    print(f"Processing {i}/{len(test_dataset)}")
                
                image = sample["image"].convert('RGB')
                true_label = sample["label"]
                
                # Make prediction
                inputs = self.processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predicted_class_id = outputs.logits.argmax().item()
                
                predictions.append(predicted_class_id)
                true_labels.append(true_label)
            
            # Calculate metrics
            self.evaluate_predictions(true_labels, predictions)
            
        except Exception as e:
            print(f"Error testing on dataset: {e}")
    
    def evaluate_predictions(self, true_labels, predictions):
        """
        Evaluate predictions and show detailed metrics
        
        Args:
            true_labels: List of true label indices
            predictions: List of predicted label indices
        """
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        # Classification report
        report = classification_report(
            true_labels, predictions, 
            target_names=self.class_names,
            digits=4
        )
        print("\nClassification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
        # Per-class accuracy
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        print("\nPer-class Accuracy:")
        for i, (class_name, acc) in enumerate(zip(self.class_names, per_class_accuracy)):
            print(f"{class_name}: {acc:.4f}")
        
        # Overall accuracy
        overall_accuracy = np.sum(cm.diagonal()) / np.sum(cm)
        print(f"\nOverall Accuracy: {overall_accuracy:.4f}")
    
    def test_directory(self, directory_path, max_images=None):
        """
        Test all images in a directory
        
        Args:
            directory_path: Path to directory containing images
            max_images: Maximum number of images to test (None for all)
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        # Find all image files
        for ext in image_extensions:
            image_files.extend(Path(directory_path).glob(f"*{ext}"))
            image_files.extend(Path(directory_path).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No image files found in {directory_path}")
            return
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"Testing {len(image_files)} images...")
        
        results = []
        for img_path in image_files:
            pred_class, confidence, _ = self.predict_single_image(
                str(img_path), show_image=False
            )
            if pred_class:
                results.append({
                    'Image': img_path.name,
                    'Predicted_Class': pred_class,
                    'Confidence': confidence
                })
        
        # Show results summary
        if results:
            df = pd.DataFrame(results)
            print("\nResults Summary:")
            print(df.to_string(index=False))
            
            # Class distribution
            print("\nPredicted Class Distribution:")
            class_counts = df['Predicted_Class'].value_counts()
            print(class_counts)
            
            return df
        
        return None

# Example usage functions
def main():
    """Main function demonstrating how to use the tester"""
    
    # Initialize the tester
    tester = CivicModelTester()
    
    print("Choose testing option:")
    print("1. Test single image")
    print("2. Test on dataset")
    print("3. Test directory of images")
    
    choice = input("Enter your choice (1-3): ")
    
    if choice == "1":
        # Test single image
        image_path = input("Enter path to image file: ")
        if os.path.exists(image_path):
            tester.predict_single_image(image_path)
        else:
            print("Image file not found!")
    
    elif choice == "2":
        # Test on dataset
        dataset_path = input("Enter dataset path (or press Enter for 'dataset'): ").strip()
        if not dataset_path:
            dataset_path = "dataset"
        tester.test_on_dataset(dataset_path)
    
    elif choice == "3":
        # Test directory
        dir_path = input("Enter directory path: ")
        if os.path.exists(dir_path):
            max_imgs = input("Max images to test (or press Enter for all): ").strip()
            max_imgs = int(max_imgs) if max_imgs.isdigit() else None
            tester.test_directory(dir_path, max_imgs)
        else:
            print("Directory not found!")
    
    else:
        print("Invalid choice!")

# Alternative: Quick test functions
def quick_test_single_image(image_path, model_path="./civic_model_final"):
    """Quick function to test a single image"""
    tester = CivicModelTester(model_path)
    return tester.predict_single_image(image_path)

def quick_test_dataset(dataset_path="dataset", model_path="./civic_model_final"):
    """Quick function to test entire dataset"""
    tester = CivicModelTester(model_path)
    tester.test_on_dataset(dataset_path)

if __name__ == "__main__":
    main()