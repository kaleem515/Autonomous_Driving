import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import os
import time
import collections

# =========================
# IMPROVEMENT 1: Reproducibility seed
# =========================
import random
random.seed(42)
np.random.seed(42)

# =========================
# PATHS & CONFIG
# =========================
BASE        = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH  = os.path.join(BASE, 'data', 'videos', 'road.mp4')
MODEL_PATH  = os.path.join(BASE, 'models', 'sign_model.keras')
OUTPUT_PATH = os.path.join(BASE, 'output', 'result.mp4')
os.makedirs(os.path.join(BASE, 'output'), exist_ok=True)

# Colors
GREEN  = (0, 210, 0)
ORANGE = (0, 165, 255)
RED    = (0, 0,   255)
YELLOW = (0, 220, 220)
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
BLUE   = (255, 100,  0)
CYAN   = (255, 220,   0)

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

VEHICLE_CLASSES = ['car','truck','bus','motorbike','person','bicycle']

# =========================
# LOAD MODELS
# =========================
print("=" * 60)
print("  AUTONOMOUS DRIVING SYSTEM  (IMPROVED VERSION)")
print("=" * 60)

yolo  = YOLO('yolov8n.pt')
model = tf.keras.models.load_model(MODEL_PATH)
print("  All models loaded!\n")

# =========================
# MODULE 1 — IMPROVED LANE DETECTION
# =========================

# IMPROVEMENT 2: Temporal Smoothing for lanes
class LaneSmoother:
    def __init__(self, alpha=0.85):
        self.alpha = alpha
        self.left  = None
        self.right = None

    def update(self, left, right):
        def smooth(prev, curr):
            if curr is None: return prev
            if prev is None: return curr
            return tuple(int(self.alpha*p + (1-self.alpha)*c)
                         for p,c in zip(prev,curr))
        self.left  = smooth(self.left,  left)
        self.right = smooth(self.right, right)
        return self.left, self.right

lane_smoother = LaneSmoother()

def detect_lanes(frame):
    h, w = frame.shape[:2]

    # IMPROVEMENT 3: CLAHE preprocessing → better edges in all lighting
    img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    gray  = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray,(5,5),0), 50, 150)

    # IMPROVEMENT 4: 4-point trapezoid ROI (was 3-point triangle)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, [np.array([
        (int(w*.05), h),
        (int(w*.42), int(h*.58)),
        (int(w*.58), int(h*.58)),
        (int(w*.95), h)
    ])], 255)
    masked = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked, 1, np.pi/180, 50,
                             minLineLength=60, maxLineGap=100)

    left_pts, right_pts = [], []
    if lines is not None:
        for l in lines:
            x1,y1,x2,y2 = l[0]
            if x2==x1: continue
            s = (y2-y1)/(x2-x1)
            if abs(s) < 0.3: continue
            (left_pts if s < 0 else right_pts).append([x1,y1,x2,y2])

    def avg_line(pts):
        if not pts: return None
        slopes = [(y2-y1)/(x2-x1) for x1,y1,x2,y2 in pts]
        intercepts = [y1-s*x1 for (x1,y1,_,_),s in zip(pts,slopes)]
        m = np.mean(slopes); b = np.mean(intercepts)
        y1_ = h; y2_ = int(h*.58)
        return (int((y1_-b)/m), y1_, int((y2_-b)/m), y2_)

    L_raw, R_raw = avg_line(left_pts), avg_line(right_pts)

    # IMPROVEMENT 2 applied: smooth lanes
    L, R = lane_smoother.update(L_raw, R_raw)

    overlay  = np.zeros_like(frame)
    detected = False

    if L:
        cv2.line(overlay,(L[0],L[1]),(L[2],L[3]),GREEN,10)
        detected = True
    if R:
        cv2.line(overlay,(R[0],R[1]),(R[2],R[3]),GREEN,10)
        detected = True
    if L and R:
        cv2.fillPoly(overlay,[np.array([[L[0],L[1]],[L[2],L[3]],
                                        [R[2],R[3]],[R[0],R[1]]])],(0,120,0))
        cx_b = (L[0]+R[0])//2; cx_t = (L[2]+R[2])//2
        cv2.line(overlay,(cx_b,h),(cx_t,int(h*.58)),YELLOW,4)

    # IMPROVEMENT 5: Lane Departure Warning
    departure = ""
    if L and R:
        lane_cx = (L[0]+R[0])//2
        offset  = (w//2) - lane_cx
        if abs(offset) > w*0.08:
            departure = "MOVE LEFT" if offset > 0 else "MOVE RIGHT"

    result = cv2.addWeighted(frame, 1, overlay, 0.4, 0)
    return result, detected, departure

# =========================
# MODULE 2 — IMPROVED OBSTACLE DETECTION
# =========================

# IMPROVEMENT 6: Track obstacle history for stability
obstacle_history = collections.deque(maxlen=5)

def detect_obstacles(frame):
    results = yolo(frame, verbose=False)
    count   = 0
    warning = False
    level   = "SAFE"

    for r in results:
        for box in r.boxes:
            name = yolo.names[int(box.cls[0])]
            conf = float(box.conf[0])

            # IMPROVEMENT 7: Higher confidence threshold → fewer false positives
            if name not in VEHICLE_CLASSES or conf < 0.50:
                continue

            count += 1
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            area  = (x2-x1)*(y2-y1)
            total = frame.shape[0]*frame.shape[1]
            ratio = area / total

            # IMPROVEMENT 8: 3-tier distance system with size
            obj_h = y2 - y1
            if ratio > 0.15 or obj_h > frame.shape[0]*0.5:
                color,dist,warning,level = RED,"VERY CLOSE",True,"BRAKE"
            elif ratio > 0.08:
                color,dist,level = ORANGE,"CLOSE","SLOW"
            else:
                color,dist = GREEN,"SAFE"

            # IMPROVEMENT 9: Cleaner bounding box labels
            label_w = max(len(f"{name} {conf:.0%}") * 12, 120)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)
            cv2.rectangle(frame,(x1,y1-32),(x1+label_w,y1),color,-1)
            cv2.putText(frame,f"{name} {conf:.0%}",
                        (x1+6,y1-8),cv2.FONT_HERSHEY_SIMPLEX,0.62,WHITE,2)
            cv2.putText(frame,dist,(x1+4,y2+24),
                        cv2.FONT_HERSHEY_SIMPLEX,0.65,color,2)

    obstacle_history.append(count)
    smooth_count = int(np.mean(obstacle_history))  # smoothed count
    return frame, smooth_count, warning, level

# =========================
# MODULE 3 — IMPROVED SIGN RECOGNITION
# =========================

# IMPROVEMENT 10: Sign prediction smoothing (voting)
sign_history = collections.deque(maxlen=5)

def detect_sign(frame):
    # IMPROVEMENT 11: CLAHE on sign input too
    img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    img  = cv2.resize(enhanced,(32,32)).astype('float32')/255.0
    pred = model.predict(np.expand_dims(img,0), verbose=0)
    cls  = int(np.argmax(pred))
    conf = float(pred[0][cls])

    # IMPROVEMENT 10: Voting — most common sign in last 5 frames
    sign_history.append(cls)
    voted_cls = max(set(sign_history), key=sign_history.count)

    return CLASS_NAMES[voted_cls], conf

# =========================
# IMPROVED HUD
# =========================
def draw_hud(frame, lane, departure, count, warn,
             sign, sconf, fc, fps, level):
    h, w = frame.shape[:2]

    # Semi-transparent top panel
    ov = frame.copy()
    cv2.rectangle(ov,(0,0),(w,145),BLACK,-1)
    cv2.addWeighted(ov,0.55,frame,0.45,0,frame)

    # Main status
    if   level=="BRAKE": st,sc = "!! EMERGENCY BRAKE REQUIRED !!",RED
    elif level=="SLOW":  st,sc = "SLOW DOWN — OBSTACLE AHEAD",    ORANGE
    elif departure:      st,sc = f"LANE DEPARTURE — {departure}",  (0,80,255)
    elif not lane:       st,sc = "LANE NOT DETECTED",              YELLOW
    else:                st,sc = "AUTONOMOUS MODE ACTIVE",         GREEN

    cv2.putText(frame, st,(20,42),cv2.FONT_HERSHEY_DUPLEX,1.0,sc,2)

    # Left info
    cv2.putText(frame,
                "Lane: " + ("DETECTED" if lane else "NOT FOUND"),
                (20,82),cv2.FONT_HERSHEY_SIMPLEX,0.72,
                GREEN if lane else RED,2)
    cv2.putText(frame,
                f"Obstacles: {count}",
                (20,118),cv2.FONT_HERSHEY_SIMPLEX,0.72,YELLOW,2)

    # Right info — sign + FPS
    if sconf > 0.70:
        cv2.putText(frame,f"Sign: {sign}",
                    (w-360,42),cv2.FONT_HERSHEY_SIMPLEX,0.72,YELLOW,2)

    cv2.putText(frame,f"FPS : {fps:.1f}",
                (w-180,82),cv2.FONT_HERSHEY_SIMPLEX,0.70,WHITE,2)
    cv2.putText(frame,f"Frame: {fc}",
                (w-180,118),cv2.FONT_HERSHEY_SIMPLEX,0.70,WHITE,2)

    # IMPROVEMENT 12: Collision warning — flashing style (odd frames only)
    if warn and fc % 2 == 0:
        cv2.putText(frame,"!! COLLISION WARNING !!",
                    (int(w*.18),int(h*.5)),
                    cv2.FONT_HERSHEY_DUPLEX,1.4,RED,4)

    # IMPROVEMENT 13: Colored status indicator bar (bottom-right corner)
    bar_color = RED if level=="BRAKE" else ORANGE if level=="SLOW" else GREEN
    cv2.rectangle(frame,(w-120,h-80),(w-10,h-10),bar_color,-1)
    cv2.putText(frame, level,
                (w-108,h-35),cv2.FONT_HERSHEY_SIMPLEX,0.65,WHITE,2)

    # Bottom bar
    cv2.rectangle(frame,(0,h-42),(w,h),BLACK,-1)
    cv2.putText(frame,
                "Autonomous Driving System — AI Computer Vision | IUB 2026",
                (20,h-14),cv2.FONT_HERSHEY_SIMPLEX,0.62,WHITE,1)

    return frame

# =========================
# MAIN LOOP
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)
w_v = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_v = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_v = cap.get(cv2.CAP_PROP_FPS) or 25

out = cv2.VideoWriter(OUTPUT_PATH,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps_v,(w_v,h_v))

fc         = 0
sign_text  = "Scanning..."
sign_conf  = 0.0
prev_time  = time.time()

# Stats tracking
total_lanes    = 0
total_warnings = 0

print("  Running... Press Q to quit\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    fc += 1
    now   = time.time()
    lfps  = 1 / max(now-prev_time, 0.001)
    prev_time = now

    frame, lane, departure       = detect_lanes(frame)
    frame, count, warn, level    = detect_obstacles(frame)

    # IMPROVEMENT 14: Sign detection every 15 frames (was 20)
    if fc % 15 == 0:
        sign_text, sign_conf = detect_sign(frame)

    frame = draw_hud(frame, lane, departure, count, warn,
                     sign_text, sign_conf, fc, lfps, level)

    if lane:    total_lanes    += 1
    if warn:    total_warnings += 1

    out.write(frame)
    cv2.imshow('Autonomous Driving System (Improved)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# IMPROVEMENT 15: Final stats print
lane_rate = (total_lanes / fc * 100) if fc > 0 else 0
print("=" * 60)
print(f"  Frames processed  : {fc}")
print(f"  Lane detect rate  : {lane_rate:.1f}%")
print(f"  Total warnings    : {total_warnings}")
print(f"  Output saved      : {OUTPUT_PATH}")
print("=" * 60)
print("  ALL IMPROVEMENTS APPLIED — READY FOR SUBMISSION!")
print("=" * 60)