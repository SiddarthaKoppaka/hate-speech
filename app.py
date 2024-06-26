from flask import Flask, render_template, request
from predictor import make_prediction
from langdetect import detect
from dotenv import load_dotenv
from huggingface_hub import HfApi, HfFolder
import os
app = Flask(__name__)


# Load environment variables from .env file (ensure this is done securely in production)
load_dotenv()

# Get the Hugging Face token from environment variables
hf_token = os.getenv('HUGGINGFACE_TOKEN')

if hf_token:
    # Log in to Hugging Face Hub
    HfFolder.save_token(hf_token)
    print("Logged into Hugging Face Hub successfully.")
else:
    print("Hugging Face token not found. Please set the HUGGINGFACE_TOKEN environment variable.")


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = "Prediction"
    if request.method == 'POST':
        # Get form data
        text_input = request.form['text_area']

        if detect(text_input) != 'te':
            prediction = "Wrong language Dude !!"
            return render_template('index.html', prediction=prediction)
        
        model_choice = request.form['model_choice']
        
        # Call your prediction function
        prediction = make_prediction(text_input, model_choice)
        # print(prediction)
    
    # Assuming 'models' is a list of model names for the dropdown
    models = ['BERT', 'DistilBERT', 'MuRIL', 'Indic-BERT', 'RoBERTa', 'NLLB', 'BART']

    # Render the template with prediction and models
    return render_template('index.html', prediction=prediction, models=models)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
