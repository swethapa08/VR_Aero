from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Adjust paths for Vercel deployment
if os.environ.get('VERCEL_ENV'):
    # Vercel environment
    template_dir = os.path.join(os.getcwd(), 'frontend', 'templates')
    static_dir = os.path.join(os.getcwd(), 'frontend', 'static')
    model_path = os.path.join(os.getcwd(), 'server', 'models', 'efficiency_model.pkl')
else:
    # Local environment
    template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend', 'templates'))
    static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend', 'static'))
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'efficiency_model.pkl')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)  # Enable CORS for frontend-backend communication

# Load the model
model = None
try:
    logger.debug(f"Resolved model path: {os.path.abspath(model_path)}")
    if not os.path.exists(model_path):
        logger.error(f"Model file does not exist at {model_path}")
        raise FileNotFoundError(f"Model file {model_path} not found")
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    logger.info("Model loaded successfully")
except FileNotFoundError as e:
    logger.error(str(e))
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/Activities')
def activities():
    return render_template('Activities.html')

@app.route('/About')
def about():
    return render_template('About.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model:
            logger.error("Prediction failed: Model not loaded")
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.get_json()
        if not data:
            logger.warning("No data provided in request")
            return jsonify({'error': 'No data provided'}), 400

        # Extract features in the correct order
        features = [
            data.get('Heart_Rate'),
            data.get('Eye_Tracking'),
            data.get('Reaction_Time'),
            data.get('Flight_Precision'),
            data.get('Decision_Speed'),
            data.get('Error_Rate')
        ]
        logger.debug(f"Received features: {features}")

        # Validate inputs
        if None in features or any(not isinstance(x, (int, float)) for x in features):
            logger.warning(f"Invalid input data: {features}")
            return jsonify({'error': 'Invalid or missing input data'}), 400

        # Prepare features for prediction
        features = np.array(features).reshape(1, -1)

        # Predict success rate
        success_rate = model.predict(features)[0]

        # Generate suggestions based on success rate
        suggestions = []
        if success_rate < 70:
            suggestions.append("Increase practice with high-stress scenarios.")
            suggestions.append("Focus on improving reaction time through drills.")
        elif success_rate < 90:
            suggestions.append("Refine flight precision with targeted simulations.")
            suggestions.append("Optimize decision-making speed in VR training.")
        else:
            suggestions.append("Maintain current training regimen.")
            suggestions.append("Explore advanced modules for further improvement.")

        logger.info(f"Prediction successful: SuccessRate={success_rate:.2f}")
        return jsonify({
            'SuccessRate': float(success_rate),
            'Suggestions': suggestions
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# Main execution block for local development
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(debug=debug, host='0.0.0.0', port=port)