"""YOLO 실시간 객체탐지 카메라.

실행:  python yolo_camera.py [--cam 0] [--model yolo11n.pt] [--conf 0.4]
종료:  q 또는 ESC / 스크린샷: s
"""
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

p = argparse.ArgumentParser()
p.add_argument("--cam", type=int, default=0)
p.add_argument("--model", default="yolo11n.pt")
p.add_argument("--conf", type=float, default=0.4)
args = p.parse_args()

out_dir = Path(__file__).parent / "captures"
out_dir.mkdir(exist_ok=True)

model = YOLO(args.model)
cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit(f"카메라 {args.cam}번을 열 수 없습니다.")

prev = time.time()
while True:
    ok, frame = cap.read()
    if not ok:
        break
    result = model(frame, conf=args.conf, verbose=False)[0]
    annotated = result.plot()

    now = time.time()
    fps = 1 / max(now - prev, 1e-6)
    prev = now
    cv2.putText(annotated, f"FPS {fps:.1f}  objects {len(result.boxes)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("YOLO Camera (q/ESC: quit, s: save)", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        break
    if key == ord("s"):
        f = out_dir / f"capture_{time.strftime('%H%M%S')}.jpg"
        cv2.imwrite(str(f), annotated)
        print("saved", f)

cap.release()
cv2.destroyAllWindows()
