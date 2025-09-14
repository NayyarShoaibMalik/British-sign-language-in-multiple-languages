This project demonstrates a machine learning pipeline to recognize British Sign Language (BSL) numbers and alphabet hand positions using this Kaggle dataset. 
The pipeline covers preprocessing raw hand landmarks, training and evaluating machine learning models, and deploying the best model via a Flask web application.
**Project Structure **

├── BSL_Dataset_Processing.py   # Preprocessing raw dataset into numerical features
├── BSL_Train_Classifier.py     # Training multiple models and saving best one
├── BSL_Testing.py              # Testing & evaluation of trained models
├── app.py                      # Flask-based web app for prediction
├── models/
│   └── best_model.pkl          # Saved trained model (pickle file)
├── static/                     # Static files for Flask app
├── templates/                  # HTML templates for Flask app
└── README.md                   # Project documentation
🚀 Workflow
1. Dataset
Source: Kaggle – BSL Numbers and Alphabet Hand Position for Mediapipe
Data contains Mediapipe landmark coordinates for hand signs representing numbers and alphabet.
2. Preprocessing (BSL_Dataset_Processing.py)
Reads raw dataset.
Cleans and transforms hand landmark positions into numerical feature vectors.
Normalization and reshaping to prepare input for ML models.
Saves processed dataset for training.
3. Model Training (BSL_Train_Classifier.py)
Trains 3–4 machine learning models (e.g., Random Forest, SVM, XGBoost, Neural Networks).
Evaluates models using accuracy and other metrics.
Selects the best performing model.
Saves it in pickle format (best_model.pkl).
4. Testing & Evaluation (BSL_Testing.py)
Loads trained models.
Runs predictions on test dataset.
Reports accuracy, confusion matrix, and performance comparison.
5. Flask Web Application (app.py)
Provides a web interface to upload or capture hand signs.
Preprocesses input using same pipeline.
Loads the best saved model (best_model.pkl).
Returns predicted letter or number as output.

