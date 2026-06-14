import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, classification_report,
    confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time

# =========================
# PATHS
# =========================
BASE       = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_KERAS = os.path.join(BASE, '..', 'models', 'sign_model.keras')
MODEL_H5    = os.path.join(BASE, '..', 'models', 'sign_model.h5')

if os.path.exists(MODEL_KERAS):
    MODEL_PATH = MODEL_KERAS
    print("Using model: sign_model.keras")
elif os.path.exists(MODEL_H5):
    MODEL_PATH = MODEL_H5
    print("Using model: sign_model.h5")
else:
    print("ERROR: No model found in models/ folder.")
    exit()

GTSRB_BASE = os.path.join(BASE, '..', 'data', 'signs', 'gtsrb_new')
possible_paths = [
    os.path.join(GTSRB_BASE, 'Test'),
    os.path.join(GTSRB_BASE, 'Train'),
    os.path.join(BASE, '..', 'data', 'signs', 'Test'),
    os.path.join(BASE, '..', 'data', 'signs', 'Train'),
]

TEST_PATH = None
for path in possible_paths:
    if os.path.exists(path):
        subfolders = [f for f in os.listdir(path)
                      if os.path.isdir(os.path.join(path,f)) and f.isdigit()]
        if len(subfolders) > 0:
            TEST_PATH = path
            print(f"Using test data : {path}")
            print(f"Found {len(subfolders)} class folders")
            break

if TEST_PATH is None:
    print("ERROR: No valid test data folder found.")
    exit()

IMG_SIZE    = 32
NUM_CLASSES = 43

CLASS_NAMES = [
    'Speed 20','Speed 30','Speed 50','Speed 60','Speed 70',
    'Speed 80','End Speed 80','Speed 100','Speed 120',
    'No Passing','No Passing 3.5t','Priority Next',
    'Priority Road','Yield','Stop','No Vehicles',
    'No Vehicles 3.5t','No Entry','General Caution',
    'Curve Left','Curve Right','Double Curve','Bumpy Road',
    'Slippery Road','Road Narrows','Road Work',
    'Traffic Signals','Pedestrians','Children Crossing',
    'Bicycles','Ice/Snow','Wild Animals','End Restrictions',
    'Turn Right','Turn Left','Ahead Only',
    'Straight or Right','Straight or Left',
    'Keep Right','Keep Left','Roundabout',
    'End No Passing','End No Passing 3.5t'
]

print("=" * 60)
print("  PROJECT EVALUATION REPORT  (IMPROVED VERSION)")
print("  Autonomous Driving Mini-System")
print("=" * 60)

# =========================
# LOAD MODEL
# =========================
print("\nLoading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded!")

# =========================
# IMPROVEMENT 1: CLAHE Preprocessing (same as training)
# Consistent preprocessing = better evaluation accuracy
# =========================
def preprocess_image(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# =========================
# LOAD TEST DATA
# =========================
def load_test_data(path):
    images, labels = [], []
    loaded_classes = 0

    for label in range(NUM_CLASSES):
        folder = os.path.join(path, str(label))
        if not os.path.exists(folder):
            continue
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.png','.jpg','.jpeg','.ppm'))]
        if not files:
            continue

        loaded_classes += 1
        for img_file in files[:50]:
            img = cv2.imread(os.path.join(folder, img_file))
            if img is None: continue
            img = preprocess_image(img)           # IMPROVEMENT 1
            img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
            images.append(img)
            labels.append(label)

    print(f"  Classes loaded : {loaded_classes}")
    print(f"  Total images   : {len(images)}")
    return np.array(images), np.array(labels)

print("\nLoading test images...")
X_test, y_true = load_test_data(TEST_PATH)

if len(X_test) == 0:
    print("ERROR: No images loaded.")
    exit()

X_test = X_test.astype('float32') / 255.0

# =========================
# PREDICTIONS
# =========================
print("\nRunning predictions...")
start_time  = time.time()
predictions = model.predict(X_test, verbose=1)
pred_time   = time.time() - start_time
y_pred      = np.argmax(predictions, axis=1)

# =========================
# METRICS
# =========================
accuracy  = accuracy_score(y_true, y_pred)  * 100
precision = precision_score(y_true, y_pred, average='weighted', zero_division=0) * 100
recall    = recall_score(y_true, y_pred,    average='weighted', zero_division=0) * 100
f1        = (2*precision*recall/(precision+recall)) if (precision+recall)>0 else 0
avg_conf  = float(np.max(predictions, axis=1).mean()) * 100

print("\n" + "=" * 60)
print("  CNN TRAFFIC SIGN RECOGNITION — RESULTS")
print("=" * 60)
print(f"  Accuracy          : {accuracy:.2f}%")
print(f"  Precision         : {precision:.2f}%")
print(f"  Recall            : {recall:.2f}%")
print(f"  F1 Score          : {f1:.2f}%")
print(f"  Avg Confidence    : {avg_conf:.2f}%")
print(f"  Prediction time   : {pred_time:.2f}s ({len(X_test)} images)")
print(f"  Speed             : {len(X_test)/pred_time:.0f} images/sec")
print("=" * 60)

# =========================
# IMPROVEMENT 2: Metrics Bar Chart (better styled)
# =========================
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Avg Conf']
values  = [accuracy, precision, recall, f1, avg_conf]
colors  = ['#27ae60','#2980b9','#c0392b','#f39c12','#8e44ad']

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(metrics, values, color=colors, edgecolor='black',
              linewidth=0.5, width=0.55)
ax.set_ylim(0, 115)
ax.set_ylabel('Score (%)', fontsize=13)
ax.set_title('CNN Traffic Sign Recognition — Evaluation Metrics\n'
             'Autonomous Driving Mini-System | IUB 2026',
             fontsize=14, fontweight='bold', pad=15)
ax.axhline(y=90, color='red', linestyle='--', alpha=0.4, label='90% threshold')
ax.legend(fontsize=11)

for bar, val in zip(bars, values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
            f'{val:.1f}%', ha='center', fontsize=12, fontweight='bold')

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
chart_path = os.path.join(OUTPUT_DIR, 'metrics_chart.png')
plt.savefig(chart_path, dpi=150)
plt.close()
print(f"\n  Metrics chart saved : {chart_path}")

# =========================
# IMPROVEMENT 3: Confusion Matrix
# Kaunsi class kahan galat classify ho rahi hai — clearly dikhta hai
# =========================
unique_labels = np.unique(y_true)

if len(unique_labels) <= 20:
    cm       = confusion_matrix(y_true, y_pred, labels=unique_labels)
    used_names = [CLASS_NAMES[i] for i in unique_labels if i < len(CLASS_NAMES)]

    fig, ax = plt.subplots(figsize=(14,12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=used_names,
                yticklabels=used_names,
                ax=ax, linewidths=0.3)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual',    fontsize=12)
    ax.set_title('Confusion Matrix — Traffic Sign Recognition',
                 fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=130)
    plt.close()
    print(f"  Confusion matrix saved : {cm_path}")

# =========================
# IMPROVEMENT 4: Top-5 Best & Worst Classes
# =========================
unique_labels = np.unique(y_true)
per_class_acc = []
for lbl in unique_labels:
    mask     = y_true == lbl
    cls_acc  = accuracy_score(y_true[mask], y_pred[mask]) * 100
    per_class_acc.append((CLASS_NAMES[lbl], cls_acc, mask.sum()))

per_class_acc.sort(key=lambda x: x[1], reverse=True)

print("\n  TOP 5 BEST CLASSIFIED SIGNS:")
for name, acc, n in per_class_acc[:5]:
    print(f"    {name:30s} -> {acc:.1f}%  ({n} samples)")

print("\n  TOP 5 WORST CLASSIFIED SIGNS:")
for name, acc, n in per_class_acc[-5:]:
    print(f"    {name:30s} -> {acc:.1f}%  ({n} samples)")

# =========================
# LANE DETECTION EVALUATION
# =========================
print("\n" + "=" * 60)
print("  LANE DETECTION EVALUATION")
print("=" * 60)

VIDEO_PATH = os.path.join(BASE, '..', 'data', 'videos', 'road.mp4')

def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [np.array([
        (int(width*.05), height),
        (int(width*.42), int(height*.58)),
        (int(width*.58), int(height*.58)),
        (int(width*.95), height)
    ])], 255)
    return cv2.bitwise_and(edges, mask)

if os.path.exists(VIDEO_PATH):
    cap             = cv2.VideoCapture(VIDEO_PATH)
    total_frames    = 0
    detected_frames = 0
    left_detected   = 0
    right_detected  = 0

    print("  Analyzing video...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        total_frames += 1

        # CLAHE preprocessing
        img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        gray   = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        blur   = cv2.GaussianBlur(gray,(5,5),0)
        edges  = cv2.Canny(blur, 50, 150)
        masked = region_of_interest(edges)
        lines  = cv2.HoughLinesP(masked,1,np.pi/180,
                                  threshold=50,
                                  minLineLength=80,
                                  maxLineGap=60)

        has_left = has_right = False
        if lines is not None:
            for l in lines:
                x1,y1,x2,y2 = l[0]
                if x2==x1: continue
                s = (y2-y1)/(x2-x1)
                if s < -0.3: has_left  = True
                if s >  0.3: has_right = True

        if has_left or has_right:
            detected_frames += 1
        if has_left:  left_detected  += 1
        if has_right: right_detected += 1

    cap.release()
    lane_rate  = (detected_frames/total_frames*100) if total_frames>0 else 0
    left_rate  = (left_detected/total_frames*100)   if total_frames>0 else 0
    right_rate = (right_detected/total_frames*100)  if total_frames>0 else 0

    print(f"  Total frames       : {total_frames}")
    print(f"  Lanes detected     : {detected_frames}")
    print(f"  Overall rate       : {lane_rate:.1f}%")
    print(f"  Left lane rate     : {left_rate:.1f}%")    # IMPROVEMENT 4: split L/R
    print(f"  Right lane rate    : {right_rate:.1f}%")
else:
    lane_rate = left_rate = right_rate = 0
    print("  Video not found — skipping lane evaluation")

# =========================
# IMPROVEMENT 5: Detailed Text Report
# =========================
used_names_report = [CLASS_NAMES[i] for i in unique_labels if i < len(CLASS_NAMES)]
report = classification_report(y_true, y_pred,
                                labels=unique_labels,
                                target_names=used_names_report,
                                zero_division=0)

report_path = os.path.join(OUTPUT_DIR, 'classification_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 65 + "\n")
    f.write("  AUTONOMOUS DRIVING SYSTEM — COMPLETE EVALUATION REPORT\n")
    f.write("  Islamia University of Bahawalpur | CS Department | 2026\n")
    f.write("=" * 65 + "\n\n")
    f.write("MODULE 1 — LANE DETECTION (OpenCV + CLAHE)\n")
    f.write("-" * 65 + "\n")
    f.write(f"  Overall Detection Rate : {lane_rate:.1f}%\n")
    f.write(f"  Left  Lane Rate        : {left_rate:.1f}%\n")
    f.write(f"  Right Lane Rate        : {right_rate:.1f}%\n\n")
    f.write("MODULE 2 — TRAFFIC SIGN CNN\n")
    f.write("-" * 65 + "\n")
    f.write(f"  Accuracy         : {accuracy:.2f}%\n")
    f.write(f"  Precision        : {precision:.2f}%\n")
    f.write(f"  Recall           : {recall:.2f}%\n")
    f.write(f"  F1 Score         : {f1:.2f}%\n")
    f.write(f"  Avg Confidence   : {avg_conf:.2f}%\n")
    f.write(f"  Prediction Speed : {len(X_test)/pred_time:.0f} images/sec\n\n")
    f.write("MODULE 3 — OBSTACLE DETECTION (YOLOv8n)\n")
    f.write("-" * 65 + "\n")
    f.write("  Model            : YOLOv8n (pretrained COCO)\n")
    f.write("  Classes          : car, truck, bus, person, bicycle, motorbike\n")
    f.write("  Conf Threshold   : 0.50\n\n")
    f.write("PER CLASS BREAKDOWN (CNN):\n")
    f.write("=" * 65 + "\n\n")
    f.write(report)
    f.write("\n\nTOP 5 BEST CLASSES:\n")
    for name,acc,n in per_class_acc[:5]:
        f.write(f"  {name:30s} -> {acc:.1f}%\n")
    f.write("\nTOP 5 WORST CLASSES:\n")
    for name,acc,n in per_class_acc[-5:]:
        f.write(f"  {name:30s} -> {acc:.1f}%\n")

print(f"\n  Report saved : {report_path}")

# =========================
# FINAL SUMMARY
# =========================
print("\n" + "=" * 60)
print("  COMPLETE PROJECT SUMMARY")
print("=" * 60)
print(f"  Module 1 — Lane Detection")
print(f"    Detection Rate     : {lane_rate:.1f}%")
print(f"    Left Lane Rate     : {left_rate:.1f}%")
print(f"    Right Lane Rate    : {right_rate:.1f}%")
print(f"\n  Module 2 — Traffic Sign CNN")
print(f"    Accuracy           : {accuracy:.2f}%")
print(f"    Precision          : {precision:.2f}%")
print(f"    Recall             : {recall:.2f}%")
print(f"    F1 Score           : {f1:.2f}%")
print(f"    Avg Confidence     : {avg_conf:.2f}%")
print(f"\n  Module 3 — Obstacle Detection")
print(f"    Model              : YOLOv8n Pretrained")
print(f"    Conf Threshold     : 0.50 (improved from 0.45)")
print("=" * 60)
print("\n  Output files saved:")
print("    metrics_chart.png")
print("    confusion_matrix.png        ← NEW")
print("    classification_report.txt")
print("    result.mp4  (from main.py)")
print("\n  EVALUATION COMPLETE — SUBMISSION READY!")
print("=" * 60)