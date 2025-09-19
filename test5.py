from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import io
import base64
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
        """Initialize the model for API use"""
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.class_names = [
            'BrokenStreetLight', 'DrainageOverFlow', 'GarbageNotOverflow', 
            'GarbageOverflow', 'NoPotHole', 'NotBrokenStreetLight', 'PotHole'
        ]
        self.load_model()
    
    def load_model(self):
        """Load the trained model and processor"""
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
        """
        Predict the class of an uploaded image
        
        Args:
            image: PIL Image object
            
        Returns:
            dict: Prediction results
        """
        try:
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Preprocess image
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                predicted_class_id = probabilities.argmax().item()
                confidence = probabilities[0][predicted_class_id].item()
            
            # Get top 3 predictions
            top_3 = torch.topk(probabilities, 3)
            top_predictions = []
            for prob, idx in zip(top_3.values[0], top_3.indices[0]):
                top_predictions.append({
                    'class': self.class_names[idx.item()],
                    'confidence': float(prob.item())
                })
            
            result = {
                'success': True,
                'predicted_class': self.class_names[predicted_class_id],
                'confidence': float(confidence),
                'top_predictions': top_predictions,
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

# Initialize the model
civic_model = CivicModelAPI()

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Civic Issue Classifier</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .container { background-color: #f5f5f5; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
        .result { background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin-top: 20px; }
        .error { background-color: #f5e8e8; padding: 15px; border-radius: 5px; margin-top: 20px; }
        .upload-btn { background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        .upload-btn:hover { background-color: #45a049; }
        #preview { max-width: 400px; max-height: 300px; margin: 10px 0; }
        .prediction-item { margin: 5px 0; padding: 5px; background-color: #fff; border-radius: 3px; }
        .confidence-bar { height: 20px; background-color: #ddd; border-radius: 3px; overflow: hidden; }
        .confidence-fill { height: 100%; background-color: #4CAF50; }
    </style>
</head>
<body>
    <h1>🏙️ Civic Issue Classifier</h1>
    <p>Upload an image to classify civic issues like potholes, broken street lights, garbage overflow, etc.</p>
    
    <div class="container">
        <h3>Upload Image</h3>
        <input type="file" id="imageInput" accept="image/*" onchange="previewImage()" />
        <br><br>
        <button class="upload-btn" onclick="uploadImage()">Classify Image</button>
        <br><br>
        <img id="preview" style="display:none;" />
    </div>
    
    <div id="result"></div>
    
    <div class="container">
        <h3>API Documentation</h3>
        <p><strong>Endpoint:</strong> POST /predict</p>
        <p><strong>Content-Type:</strong> multipart/form-data</p>
        <p><strong>Parameter:</strong> image (file)</p>
        <p><strong>Response:</strong> JSON with prediction results</p>
        
        <h4>Example using curl:</h4>
        <code>curl -X POST -F "image=@your_image.jpg" http://localhost:5000/predict</code>
        
        <h4>Supported Classes:</h4>
        <ul>
            <li>BrokenStreetLight</li>
            <li>DrainageOverFlow</li>
            <li>GarbageNotOverflow</li>
            <li>GarbageOverflow</li>
            <li>NoPotHole</li>
            <li>NotBrokenStreetLight</li>
            <li>PotHole</li>
        </ul>
    </div>

    <script>
        function previewImage() {
            const input = document.getElementById('imageInput');
            const preview = document.getElementById('preview');
            
            if (input.files && input.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                };
                reader.readAsDataURL(input.files[0]);
            }
        }
        
        function uploadImage() {
            const input = document.getElementById('imageInput');
            const resultDiv = document.getElementById('result');
            
            if (!input.files || !input.files[0]) {
                alert('Please select an image first!');
                return;
            }
            
            const formData = new FormData();
            formData.append('image', input.files[0]);
            
            resultDiv.innerHTML = '<div class="container">Processing...</div>';
            
            fetch('/predict', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                displayResult(data);
            })
            .catch(error => {
                console.error('Error:', error);
                resultDiv.innerHTML = '<div class="error">Error: ' + error.message + '</div>';
            });
        }
        
        function displayResult(data) {
            const resultDiv = document.getElementById('result');
            
            if (data.success) {
                let html = '<div class="result">';
                html += '<h3>🎯 Prediction Results</h3>';
                html += '<h4>Main Prediction: ' + data.predicted_class + '</h4>';
                html += '<p>Confidence: ' + (data.confidence * 100).toFixed(2) + '%</p>';
                
                html += '<h4>All Predictions:</h4>';
                data.top_predictions.forEach(pred => {
                    const percentage = (pred.confidence * 100).toFixed(2);
                    html += '<div class="prediction-item">';
                    html += '<strong>' + pred.class + ':</strong> ' + percentage + '%';
                    html += '<div class="confidence-bar">';
                    html += '<div class="confidence-fill" style="width: ' + percentage + '%"></div>';
                    html += '</div></div>';
                });
                
                html += '<p><small>Timestamp: ' + data.timestamp + '</small></p>';
                html += '</div>';
                resultDiv.innerHTML = html;
            } else {
                resultDiv.innerHTML = '<div class="error">Error: ' + data.error + '</div>';
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Serve the web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for image prediction
    
    Expects:
        - POST request with multipart/form-data
        - 'image' file in the request
    
    Returns:
        - JSON response with prediction results
    """
    try:
        # Check if image is in request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        file = request.files['image']
        
        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No image file selected',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Check file extension
        allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in allowed_extensions:
            return jsonify({
                'success': False,
                'error': f'Unsupported file format. Allowed: {", ".join(allowed_extensions)}',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Make prediction
        result = civic_model.predict_image(image)
        
        # Return appropriate response
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': civic_model.model is not None,
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get available classes"""
    return jsonify({
        'classes': civic_model.class_names,
        'total_classes': len(civic_model.class_names),
        'timestamp': datetime.now().isoformat()
    }), 200

if __name__ == '__main__':
    # Check if model is loaded
    if civic_model.model is None:
        print("❌ Model failed to load. Please check the model path.")
        print("Make sure your trained model is available at './civic_model_final'")
    else:
        print("✅ Model loaded successfully!")
        print("🚀 Starting Flask API server...")
        print("📱 Web interface: http://localhost:5000")
        print("🔗 API endpoint: http://localhost:5000/predict")
        
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
