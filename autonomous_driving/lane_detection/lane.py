import cv2
import numpy as np
import os
import time

# =========================
# PATHS
# =========================
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(BASE, 'data', 'videos', 'road.mp4')
OUT_PATH   = os.path.join(BASE, 'output', 'lane_output.mp4')
os.makedirs(os.path.join(BASE, 'output'), exist_ok=True)

print("=" * 60)
print("  LANE DETECTION MODULE  (IMPROVED VERSION)")
print("=" * 60)
print(f"  Video path : {VIDEO_PATH}")
print(f"  Exists     : {os.path.exists(VIDEO_PATH)}")
print("=" * 60)

if not os.path.exists(VIDEO_PATH):
    print("ERROR: Video not found at data/videos/road.mp4")
    exit()

# =========================
# IMPROVEMENT 1: Adaptive ROI
# Fixed triangle ki jagah — frame size ke hisaab se auto adjust hoti hai
# =========================
def region_of_interest(edges, frame_shape):
    height, width = frame_shape[:2]
    mask = np.zeros_like(edges)

    # IMPROVEMENT: Dynamic ROI — 4-point trapezoid (better than triangle)
    polygon = np.array([[
        (int(width * 0.05), height),           # bottom-left
        (int(width * 0.42), int(height * 0.58)),  # top-left
        (int(width * 0.58), int(height * 0.58)),  # top-right
        (int(width * 0.95), height),           # bottom-right
    ]])
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)

# =========================
# IMPROVEMENT 2: Averaged Lane Lines
# Raw lines ki jagah smooth averaged lines — left aur right alag
# =========================
def average_lines(frame, lines):
    left_lines  = []
    right_lines = []
    h = frame.shape[0]

    if lines is None:
        return None, None

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)

        # Filter weak slopes
        if abs(slope) < 0.3:
            continue

        if slope < 0:
            left_lines.append(line[0])
        else:
            right_lines.append(line[0])

    def make_line(pts):
        if not pts:
            return None
        slopes = [(y2-y1)/(x2-x1) for x1,y1,x2,y2 in pts]
        intercepts = [y1 - s*x1 for (x1,y1,_,_), s in zip(pts, slopes)]
        m = np.mean(slopes)
        b = np.mean(intercepts)
        y1_l = h
        y2_l = int(h * 0.58)
        x1_l = int((y1_l - b) / m)
        x2_l = int((y2_l - b) / m)
        return (x1_l, y1_l, x2_l, y2_l)

    return make_line(left_lines), make_line(right_lines)

# =========================
# IMPROVEMENT 3: Lane Smoothing (Temporal)
# Frame-to-frame jitter kam karta hai — lines stable dikhti hain
# =========================
class LaneSmoother:
    def __init__(self, alpha=0.85):
        self.alpha   = alpha   # smoothing factor
        self.left    = None
        self.right   = None

    def smooth(self, left, right):
        def update(prev, curr):
            if curr is None:
                return prev
            if prev is None:
                return curr
            return tuple(int(self.alpha*p + (1-self.alpha)*c)
                         for p, c in zip(prev, curr))

        self.left  = update(self.left,  left)
        self.right = update(self.right, right)
        return self.left, self.right

smoother = LaneSmoother(alpha=0.85)

# =========================
# IMPROVEMENT 4: Lane Departure Warning
# Center se kitna dur hai car — warning deta hai
# =========================
def check_departure(frame, left, right):
    h, w = frame.shape[:2]
    warning = ""

    if left and right:
        lane_center = (left[0] + right[0]) // 2
        car_center  = w // 2
        offset      = car_center - lane_center

        if abs(offset) > w * 0.08:
            if offset > 0:
                warning = "LANE DEPARTURE — MOVE LEFT"
            else:
                warning = "LANE DEPARTURE — MOVE RIGHT"

    return warning

# =========================
# PROCESS FRAME
# =========================
def process_frame(frame):
    h, w = frame.shape[:2]

    # Step 1 — IMPROVEMENT 5: Better preprocessing
    # YUV color space → better edge detection in different lighting
    img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    gray  = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Step 2 — ROI
    masked = region_of_interest(edges, frame.shape)

    # Step 3 — Hough Lines
    lines = cv2.HoughLinesP(
        masked,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=80,
        maxLineGap=60
    )

    # Step 4 — Average + Smooth
    left_raw, right_raw    = average_lines(frame, lines)
    left_avg, right_avg    = smoother.smooth(left_raw, right_raw)

    # Step 5 — Draw overlay
    overlay = np.zeros_like(frame)
    detected = False

    if left_avg:
        x1,y1,x2,y2 = left_avg
        cv2.line(overlay, (x1,y1), (x2,y2), (0, 210, 0), 10)
        detected = True
    if right_avg:
        x1,y1,x2,y2 = right_avg
        cv2.line(overlay, (x1,y1), (x2,y2), (0, 210, 0), 10)
        detected = True

    # IMPROVEMENT 6: Lane Fill + Center Line
    if left_avg and right_avg:
        lx1,ly1,lx2,ly2 = left_avg
        rx1,ry1,rx2,ry2 = right_avg
        pts = np.array([[lx1,ly1],[lx2,ly2],[rx2,ry2],[rx1,ry1]])
        cv2.fillPoly(overlay, [pts], (0, 100, 0))
        # Yellow center line
        cx_b = (lx1+rx1)//2
        cx_t = (lx2+rx2)//2
        cv2.line(overlay, (cx_b,ly1), (cx_t,ly2), (0,220,220), 3)

    result = cv2.addWeighted(frame, 1.0, overlay, 0.4, 0)

    # Step 6 — Lane departure check
    departure_warning = check_departure(frame, left_avg, right_avg)

    # Step 7 — HUD
    status       = "Lane: DETECTED" if detected else "Lane: NOT DETECTED"
    status_color = (0, 210, 0)      if detected else (0, 0, 255)

    # Dark top bar
    bar = result.copy()
    cv2.rectangle(bar, (0,0), (w, 100), (0,0,0), -1)
    cv2.addWeighted(bar, 0.5, result, 0.5, 0, result)

    cv2.putText(result, status,
                (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1.0, status_color, 2)

    if departure_warning:
        cv2.putText(result, departure_warning,
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 80, 255), 2)

    # Bottom bar
    cv2.rectangle(result, (0, h-40), (w, h), (0,0,0), -1)
    cv2.putText(result,
                "Lane Detection Module — Autonomous Driving System",
                (20, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    return result, detected

# =========================
# RUN ON VIDEO
# =========================
print("\nStarting improved lane detection...")
print("Press Q to quit\n")

cap   = cv2.VideoCapture(VIDEO_PATH)
w_v   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_v   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps_v = cap.get(cv2.CAP_PROP_FPS) or 25

out = cv2.VideoWriter(OUT_PATH,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps_v, (w_v, h_v))

frame_count   = 0
detected_count = 0
prev_time     = time.time()

if not cap.isOpened():
    print("ERROR: Cannot open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        break

    frame_count += 1
    output, detected = process_frame(frame)

    if detected:
        detected_count += 1

    # FPS
    now  = time.time()
    lfps = 1 / max(now - prev_time, 0.001)
    prev_time = now

    cv2.putText(output,
                f"Frame: {frame_count}  |  FPS: {lfps:.1f}",
                (20, h_v - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    out.write(output)
    cv2.imshow('Lane Detection (Improved) — Press Q', output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped by user.")
        break

cap.release()
out.release()
cv2.destroyAllWindows()

rate = (detected_count / frame_count * 100) if frame_count > 0 else 0
print(f"\n  Frames processed : {frame_count}")
print(f"  Lanes detected   : {detected_count}")
print(f"  Detection rate   : {rate:.1f}%")
print(f"  Output saved     : {OUT_PATH}")
print("=" * 60)
print("  All improvements applied! Run main.py for full system.")
print("=" * 60)