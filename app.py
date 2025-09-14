from flask import Flask, render_template, Response, jsonify, session, request
import cv2 as cv
import mediapipe as mp
import numpy as np
import pickle
import time
import pyttsx3
import warnings
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from googletrans import Translator

app = Flask(__name__)
app.secret_key = 'your_super_secret_key_123'
translator = Translator()

warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.base")

# Load the trained BSL Model
with open('model_BSL.pkl', 'rb') as file:
    model_bsl_dict = pickle.load(file)
model_bsl = model_bsl_dict['model']

def make_letter_map(classes):
    mapping = {}
    for cls in classes:
        if str(cls).isdigit():
            mapping[cls] = chr(ord("A") + int(cls))
        elif isinstance(cls, (bytes, str)) and len(str(cls)) == 1:
            mapping[cls] = str(cls)[-1].upper()
        elif isinstance(cls, str) and cls[0].isalpha():
            mapping[cls] = cls[0].upper()
        else:
            mapping[cls] = "?"
    return mapping

labels_dict = make_letter_map(model_bsl.classes_)

# Load Emotion Detection Model 
emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

emotion_model = models.resnet18(weights=None)
emotion_model.fc = torch.nn.Linear(emotion_model.fc.in_features, 7)
full_state_dict = torch.load('resnet18-5c106cde.pth', map_location=torch.device('cpu'), weights_only=False)
filtered_state_dict = {k: v for k, v in full_state_dict.items() if not k.startswith('fc.')}
emotion_model.load_state_dict(filtered_state_dict, strict=False)
emotion_model.eval()

emotion_transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# MediaPipe Hands 
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.3)

# 
predicted_text = ""
current_model = None
current_labels_dict = None

@app.route('/')
def bsl_page():
    global current_model, current_labels_dict
    current_model = model_bsl
    current_labels_dict = labels_dict
    return render_template('index.html', language="BRITISH SIGN LANGUAGE")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def get_mac_builtin_camera():
    # Try index 0 first (usually built-in camera on Mac)
    cap = cv.VideoCapture(0, cv.CAP_AVFOUNDATION)
    if cap.isOpened():
        # Additional check to verify it's not an external camera
        ret, frame = cap.read()
        if ret:
            # Built-in MacBook cameras typically have these resolutions
            common_resolutions = [(1280, 720), (640, 480)]
            if (frame.shape[1], frame.shape[0]) in common_resolutions:
                return cap
        cap.release()
    
    # If index 0 didn't work, try other indices
    for i in range(1, 5):
        cap = cv.VideoCapture(i, cv.CAP_AVFOUNDATION)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Skip if resolution is too high (likely iPhone)
                if frame.shape[1] <= 1280 and frame.shape[0] <= 720:
                    return cap
            cap.release()
    
    # Fallback to default if nothing else works
    return cv.VideoCapture(0)

def generate_frames():
    global predicted_text, current_model, current_labels_dict

    cap = get_mac_builtin_camera()
    if not cap.isOpened():
        print("No camera available.")
        # Return a black frame with error message
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv.putText(frame, "Camera Not Available", (50, 240), 
                  cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return

    prev_sign = None
    start_time = None
    detection_threshold = 1

    while cap.isOpened():
        data_aux, x_, y_ = [], [], []
        ret, frame = cap.read()
        if not ret:
            break

        H, W = frame.shape[:2]
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                for lm in hand_landmarks.landmark:
                    data_aux.extend([lm.x, lm.y])
                    x_.append(lm.x)
                    y_.append(lm.y)

            x1, y1 = int(min(x_) * W), int(min(y_) * H)
            x2, y2 = int(max(x_) * W), int(max(y_) * H)

            # ── Clamp bounding box ──
            max_width = 300
            max_height = 300
            width = min(x2 - x1, max_width)
            height = min(y2 - y1, max_height)
            x2 = x1 + width
            y2 = y1 + height

            data_aux = (data_aux + [0] * 84)[:84]
            prediction = current_model.predict([np.asarray(data_aux)])
            raw_label = prediction[0]
            predicted_character = current_labels_dict.get(raw_label, '?')

            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv.putText(frame, predicted_character, (x1, y1 - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv.LINE_AA)

            if predicted_character == prev_sign:
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time >= detection_threshold:
                    predicted_text += predicted_character
                    start_time = None
            else:
                prev_sign = predicted_character
                start_time = None

        # Emotion detection 
        try:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                face_pil = Image.fromarray(cv.cvtColor(face_img, cv.COLOR_BGR2RGB))
                input_tensor = emotion_transform(face_pil).unsqueeze(0)

                with torch.no_grad():
                    outputs = emotion_model(input_tensor)
                    _, predicted = torch.max(outputs, 1)
                    emotion = emotion_classes[predicted.item()]

                cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 2)
                cv.putText(frame, f'Emotion: {emotion}', (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                break
        except Exception as e:
            print(f"Emotion detection error: {e}")

        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv.destroyAllWindows()

# Text Control Routes 
@app.route('/clear_last_character', methods=['POST'])
def clear_last_character():
    global predicted_text
    predicted_text = predicted_text[:-1] if predicted_text else ""
    return jsonify(predicted_text=predicted_text)

@app.route('/speak_sentence', methods=['POST'])
def speak_sentence():
    engine = pyttsx3.init()
    engine.say(predicted_text)
    engine.runAndWait()
    return jsonify(success=True)

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    global predicted_text
    predicted_text = ""
    return jsonify(success=True)

@app.route('/get_predicted_text', methods=['GET'])
def get_predicted_text():
    lang = request.args.get('lang', 'en')
    try:
        if lang != 'en' and predicted_text:
            translated = translator.translate(predicted_text, dest=lang).text
            return jsonify(predicted_text=translated)
    except Exception as e:
        print(f"Translation error: {e}")
    return jsonify(predicted_text=predicted_text)

@app.route('/add_space', methods=['POST'])
def add_space():
    global predicted_text
    predicted_text += " "
    return jsonify(predicted_text=predicted_text)

@app.route('/set_language', methods=['POST'])
def set_language():
    data = request.get_json()
    if data and 'language' in data:
        session['language'] = data['language']
        return jsonify(success=True)
    return jsonify(success=False), 400

if __name__ == '__main__':
    app.run(debug=True)