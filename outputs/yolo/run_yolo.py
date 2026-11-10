"""yolo26n / yolo26n-seg 를 conf 0.25, 0.50 으로 돌려 결과를 비교 저장한다.

입력 두 개를 각각 처리한다.
  folder_bus : 11.10/bus.jpg
  sample_bus : ultralytics 패키지 내장 샘플 assets/bus.jpg
실행:  python run_yolo.py        저장: 이 스크립트가 있는 폴더(outputs/yolo/<입력이름>/)
"""
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # 11.10
MODELS = ["yolo26n.pt", "yolo26n-seg.pt"]
CONFS = [0.25, 0.50]
IMGSZ, DEVICE = 640, "cpu"
INPUTS = {
    "folder_bus": ROOT / "bus.jpg",
    "sample_bus": Path(ultralytics.__file__).parent / "assets" / "bus.jpg",
    "nk_compare": ROOT / "비교NK.png",
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


loaded = {m: YOLO(str(ROOT / m)) for m in MODELS}
inputs_meta, all_runs = {}, {}

for label, src in INPUTS.items():
    out = HERE / label
    for d in ("input", "images", "json", "masks"):
        (out / d).mkdir(parents=True, exist_ok=True)
    inp = out / "input" / src.name
    shutil.copy2(src, inp)
    img = cv2.imread(str(inp))
    assert img is not None, f"이미지를 읽을 수 없음: {inp}"
    inputs_meta[label] = {"source": str(src), "copy": f"{label}/input/{src.name}",
                          "sha256": sha(inp), "width": img.shape[1], "height": img.shape[0]}

    runs = []
    for mname, model in loaded.items():
        tag_m = Path(mname).stem
        for conf in CONFS:
            tag = f"{tag_m}_conf{conf:.2f}"
            r = model.predict(img, imgsz=IMGSZ, device=DEVICE, conf=conf, verbose=False)[0]
            cv2.imwrite(str(out / "images" / f"{tag}.jpg"), r.plot())

            dets = []
            for i, b in enumerate(r.boxes):
                c = int(b.cls)
                dets.append({"id": i, "class_id": c, "class_name": r.names[c],
                             "score": round(float(b.conf), 4),
                             "xyxy": [round(float(v), 1) for v in b.xyxy[0]]})
            (out / "json" / f"{tag}.json").write_text(
                json.dumps({"model": mname, "conf": conf, "imgsz": IMGSZ, "image": inp.name,
                            "n": len(dets), "detections": dets}, ensure_ascii=False, indent=2),
                encoding="utf-8")

            n_masks = 0
            if r.masks is not None:
                h, w = img.shape[:2]
                md = out / "masks" / tag
                md.mkdir(exist_ok=True)
                union = np.zeros((h, w), np.uint8)
                for i, poly in enumerate(r.masks.xy):
                    m = np.zeros((h, w), np.uint8)
                    cv2.fillPoly(m, [poly.astype(np.int32)], 255)
                    cv2.imwrite(str(md / f"{i:02d}_{dets[i]['class_name']}.png"), m)
                    union |= m
                    n_masks += 1
                cv2.imwrite(str(md / "all_union.png"), union)

            runs.append({"model": mname, "conf": conf, "n_boxes": len(dets), "n_masks": n_masks,
                         "min_score": min((d["score"] for d in dets), default=None),
                         "image": f"{label}/images/{tag}.jpg", "json": f"{label}/json/{tag}.json",
                         "masks_dir": f"{label}/masks/{tag}" if n_masks else None,
                         "speed_ms": {k: round(v, 1) for k, v in r.speed.items()}})
            print(label, tag, "boxes", len(dets), "masks", n_masks)
    all_runs[label] = runs

    # 2x2 비교 시트 (행: 모델, 열: conf)
    rows = []
    for mname in MODELS:
        row = []
        for conf in CONFS:
            im = cv2.imread(str(out / "images" / f"{Path(mname).stem}_conf{conf:.2f}.jpg"))
            cv2.rectangle(im, (0, 0), (im.shape[1], 36), (0, 0, 0), -1)
            cv2.putText(im, f"{label} | {Path(mname).stem} conf={conf:.2f}", (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            row.append(im)
        rows.append(np.hstack(row))
    cv2.imwrite(str(out / "comparison.jpg"), np.vstack(rows))

# 입력 간 비교: folder_bus 기준으로 같은 파일인지, 탐지 결과가 같은지 기록
def dets_of(label, name):
    return json.loads((HERE / label / "json" / name).read_text(encoding="utf-8"))["detections"]


same_results = {}
for label in INPUTS:
    same_results[label] = {
        "same_file_as_folder_bus": inputs_meta[label]["sha256"] == inputs_meta["folder_bus"]["sha256"],
        "same_detections_as_folder_bus": all(
            dets_of(label, Path(r["json"]).name) == dets_of("folder_bus", Path(r["json"]).name)
            for r in all_runs[label]),
    }

# 입력 간 비교 시트: 행=모델, 열=입력 (conf 0.25 / 0.50 각각)
for conf in CONFS:
    rows = []
    for mname in MODELS:
        row = []
        for label in INPUTS:
            im = cv2.imread(str(HERE / label / "images" / f"{Path(mname).stem}_conf{conf:.2f}.jpg"))
            im = cv2.resize(im, (int(im.shape[1] * 400 / im.shape[0]), 400))
            cv2.rectangle(im, (0, 0), (im.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(im, f"{label} | {Path(mname).stem} conf={conf:.2f}", (6, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            row.append(im)
        rows.append(row)
    wmax = max(sum(i.shape[1] for i in r) for r in rows)
    rows = [np.hstack(r + [np.zeros((400, wmax - sum(i.shape[1] for i in r), 3), np.uint8)]) for r in rows]
    cv2.imwrite(str(HERE / f"comparison_inputs_conf{conf:.2f}.jpg"), np.vstack(rows))

manifest = {
    "created": datetime.now().astimezone().isoformat(timespec="seconds"),
    "inputs": inputs_meta,
    "input_comparison_vs_folder_bus": same_results,
    "settings": {"imgsz": IMGSZ, "device": DEVICE, "confs": CONFS, "models": MODELS},
    "models": {m: {"file": str(ROOT / m), "sha256": sha(ROOT / m)} for m in MODELS},
    "packages": {"python": platform.python_version(), "ultralytics": version("ultralytics"),
                 "torch": torch.__version__, "opencv-python": cv2.__version__,
                 "numpy": np.__version__},
    "execution": {"script": str(Path(__file__).resolve()), "cwd": str(Path.cwd()),
                  "python_executable": sys.executable, "platform": platform.platform(),
                  "machine": platform.machine(), "cuda_available": torch.cuda.is_available()},
    "runs": all_runs,
}
(HERE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(same_results, ensure_ascii=False))
print("manifest.json 저장 완료")
