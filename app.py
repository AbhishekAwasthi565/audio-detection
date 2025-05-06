import os
from flask import Flask, render_template, request, jsonify
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from joblib import load
import IPython.display as ipd

app = Flask(__name__)


UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


model = load_model('model.h5')
labelencoder = load('labelencoder.joblib')

def predict_audio(filename):
    try:
       
        audio, sample_rate = librosa.load(filename)
        
       
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=50)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
        
        
        predicted_label = np.argmax(model.predict(mfccs_scaled_features), axis=1)
        prediction_class = labelencoder.inverse_transform(predicted_label)
        
        return prediction_class[0]
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['audio_file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file:
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            
            prediction = predict_audio(filepath)
            
            if prediction is not None:
                return jsonify({
                    'prediction': prediction,
                    'audio_file': filepath
                })
            else:
                return jsonify({'error': 'Prediction failed'}), 500
                
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)