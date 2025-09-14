import cv2 as cv                     
import mediapipe as mp                
import pickle                         
import numpy as np                    
import warnings                       


warnings.filterwarnings(
    "ignore", category=UserWarning, module="google.protobuf.symbol_database"
)


MODEL_PATH = "/Users/maliknayyar/Downloads/fyp_final/model_BSL.pkl"
model = pickle.load(open(MODEL_PATH, "rb"))["model"]

# Build a robust label‑to‑letter mapping 
def make_letter_map(classes):
    mapping = {}
    for cls in classes:
        # integers 0‑25
        if str(cls).isdigit():
            mapping[cls] = chr(ord("A") + int(cls))
        # one‑character bytes/strings
        elif isinstance(cls, (bytes, str)) and len(str(cls)) == 1:
            mapping[cls] = str(cls)[-1].upper()
        # strings with a leading letter
        elif isinstance(cls, str) and cls[0].isalpha():
            mapping[cls] = cls[0].upper()
        else:
            mapping[cls] = "?"
    return mapping

LABELS_DICT = make_letter_map(model.classes_)
UNKNOWN_CHAR = "?"

print("Model classes  ➜", model.classes_)
print("Letter mapping ➜", LABELS_DICT)


def open_internal_camera():
    for idx in (0, 1, 2, 3):
        cap = cv.VideoCapture(idx, cv.CAP_AVFOUNDATION)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"[INFO] Using camera index {idx}")
                return cap
            cap.release()
    raise RuntimeError(
        "No working camera found"
    )

cap = open_internal_camera()

#  MediaPipe Hands detector 
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3,
)

# Main live‑loop 
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Empty frame – camera disconnected?")
            break

        H, W = frame.shape[:2]
        data_aux, x_, y_ = [], [], []

        # Hand detection 
        results = hands.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hlms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hlms,
                    mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
                for lm in hlms.landmark:
                    data_aux.extend([lm.x, lm.y])
                    x_.append(lm.x)
                    y_.append(lm.y)

            #  Bounding box (for text & rectangle)
            x1, y1 = int(min(x_) * W), int(min(y_) * H)
            x2, y2 = int(max(x_) * W), int(max(y_) * H)

            #  Pad / trim feature vector → exactly 84 floats
            data_aux = (data_aux + [0] * 84)[:84]

            #  Prediction 
            pred_raw = model.predict([np.asarray(data_aux)])[0]
            pred_chr = LABELS_DICT.get(pred_raw, UNKNOWN_CHAR)

            #  Draw UI 
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv.putText(
                frame,
                pred_chr,
                (x1, y1 - 10),
                cv.FONT_HERSHEY_SIMPLEX,
                1.3,
                (0, 255, 0),
                3,
                cv.LINE_AA,
            )

        #  Show frame 
        cv.imshow("BSL Live (Webcam)", frame)
        if cv.waitKey(1) & 0xFF in (ord("q"), 27):  # q  or  Esc
            break
finally:
    cap.release()
    cv.destroyAllWindows()
