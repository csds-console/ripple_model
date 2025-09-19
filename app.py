from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import io
import os
from datetime import datetime
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class CivicModelAPI:
    def __init__(self, model_path="./civic_model_final"):
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.class_names = [
            'BrokenStreetLight', 'DrainageOverFlow', 'GarbageNotOverflow',
            'GarbageOverflow', 'NoPotHole', 'NotBrokenStreetLight', 'PotHole'
        ]
        self.load_model()
    
    def load_model(self):
        try:
            logger.info(f"Loading model from {self.model_path}...")
            self.model = ViTForImageClassification.from_pretrained(self.model_path)
            self.processor = ViTImageProcessor.from_pretrained(self.model_path)
            self.model.eval()
            logger.info("Model loaded successfully!")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def predict_image(self, image):
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_class_id = probabilities.argmax().item()
                confidence = probabilities[0][predicted_class_id].item()

            top_3 = torch.topk(probabilities, 3)
            top_predictions = []
            for prob, idx in zip(top_3.values[0], top_3.indices[0]):
                top_predictions.append({
                    'class': self.class_names[idx.item()],
                    'confidence': float(prob.item())
                })

            return {
                'success': True,
                'predicted_class': self.class_names[predicted_class_id],
                'confidence': float(confidence),
                'top_predictions': top_predictions,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

civic_model = CivicModelAPI()

@app.route('/')
def home():
    return jsonify({"message": "Civic Issue Classifier API is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image selected'}), 400
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
        ext = os.path.splitext(file.filename.lower())[1]
        if ext not in allowed_extensions:
            return jsonify({'success': False, 'error': 'Unsupported file format'}), 400
        image = Image.open(io.BytesIO(file.read()))
        result = civic_model.predict_image(image)
        return jsonify(result), 200 if result['success'] else 500
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': civic_model.model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify({
        'classes': civic_model.class_names,
        'total_classes': len(civic_model.class_names),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # ✅ Use Render’s assigned port
    app.run(host='0.0.0.0', port=port, debug=False)
