# 선택 과제 A — YOLO 객체탐지·세그멘테이션 서비스

`yolo26n`(박스)과 `yolo26n-seg`(박스+마스크)를 같은 입력에 돌려, **신뢰도 문턱(conf 0.25 / 0.50)** 과 **입력 이미지**에 따라 결과가 어떻게 달라지는지 비교하고 파일로 저장한다.

- 실시간 카메라: `yolo_camera.py` (창 표시용, 결과 저장은 `s` 키 스크린샷)
- 비교 실행·저장: `outputs/yolo/run_yolo.py` ← **과제 결과물은 이쪽**

## 1. 실행 방법

```powershell
# 저장소 루트(= 이 README가 있는 폴더)에서
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python outputs\yolo\run_yolo.py
```

- 모델 가중치 `yolo26n.pt`, `yolo26n-seg.pt`는 저장소 루트에 포함되어 있다. 없으면 ultralytics가 첫 실행 때 내려받는다.
- 입력 3개를 순서대로 처리한다. 이 PC(CPU)에서 모델 추론은 한 장당 약 90~180ms였다(`manifest.json`의 `speed_ms`, 1회 측정).
- 설정은 `run_yolo.py` 상단 상수: `imgsz=640`, `device=cpu`, `conf=[0.25, 0.50]`.

## 2. 실행 위치(환경)

실제 실행 기록은 `outputs/yolo/manifest.json`의 `execution`, `packages`에 있다.

| 항목 | 값 |
|---|---|
| OS | Windows 11 Pro (10.0.26300), 로컬 PC |
| Python | 3.15.1 |
| ultralytics / torch / opencv-python / numpy | 8.4.208 / 2.15.0+cpu / 5.0.0 / 2.5.4 |
| 장치 | CPU (CUDA 사용 안 함) |
| 가상환경 | `C:\Users\quant\venvs\yolo` (OneDrive 밖) |

## 3. 입력

| 이름 | 파일 | 크기 | sha256(앞 12자) |
|---|---|---|---|
| `folder_bus` | `bus.jpg` (11.10 폴더) | 810×1080 | c02019c4979c |
| `sample_bus` | ultralytics 패키지 내장 `assets/bus.jpg` | 810×1080 | c02019c4979c |
| `nk_compare` | `비교NK.png` | 1248×699 | a06dd32a3572 |

`folder_bus`와 `sample_bus`는 **바이트 단위로 같은 파일**이다(sha256 동일). 그래서 두 입력의 탐지 결과도 완전히 같고, 둘은 재현성 확인용이다. 실질적인 비교는 **버스 사진 vs 비교NK**다.

> **저장소에서 제외한 파일**: `비교NK.png`는 방송 화면 캡처로 보여 저작권 문제로 올리지 않았다. 이 이미지에서 나온 파생물(`nk_compare/`의 입력 사본·결과 이미지·마스크·비교 시트, 입력 3개 비교 시트 `comparison_inputs_conf*.jpg`)도 제외했다. 박스 좌표 JSON(`nk_compare/json/`)과 `manifest.json`의 기록은 남아 있다. 이 입력을 재현하려면 같은 해상도(1248×699)의 이미지를 `11.10/비교NK.png`로 두고 실행하면 되며, 원본이 같은지는 `manifest.json`의 sha256(`a06dd32a…`)으로 확인할 수 있다. 아래 4장의 비교NK 수치는 이 JSON에서 나온 값이다.

## 4. 결과 (conf에 따른 박스 수)

| 입력 | yolo26n 0.25 | yolo26n 0.50 | yolo26n-seg 0.25 | yolo26n-seg 0.50 |
|---|---|---|---|---|
| 버스 사진 | 5 | 5 | 5 (마스크 5) | 5 (마스크 5) |
| 비교NK | 7 | 6 | 7 (마스크 7) | 5 (마스크 5) |

- **버스 사진**: 탐지된 5개(bus 1, person 4)의 점수가 모두 0.648 이상이라 문턱을 올려도 변화가 없다.
- **비교NK**: 점수가 낮은 객체가 있어 문턱의 영향이 보인다. 탐지 클래스는 `boat`, `person`뿐이다.
  - yolo26n: 0.4185 person 1건이 0.50에서 사라진다.
  - yolo26n-seg: 0.447, 0.3014 person 2건이 0.50에서 사라진다(육안으로 확인한 것은 하단 좌측 패널의 오른쪽 인물 1명, 나머지 1건의 위치는 확인하지 않음).
  - 두 모델은 같은 `boat`도 박스·점수가 다르다(예: 하단 군함 0.73 vs 0.70).
- 이미지의 연기·불꽃·미사일은 COCO 80클래스에 없어 탐지 대상이 아니다.

## 5. 저장 구조 (`outputs/yolo/`)

```
manifest.json                      입력 sha256, 모델·패키지 버전, 실행 위치, 설정, 실행별 결과
run_yolo.py                        실행 코드
comparison_inputs_conf0.25.jpg     입력 3개 × 모델 2개 비교 시트 (conf 0.25)
comparison_inputs_conf0.50.jpg     같은 시트 (conf 0.50)
<입력이름>/                        folder_bus, sample_bus, nk_compare
  input/                           입력 사본
  images/<모델>_conf<값>.jpg       박스·클래스·점수(+마스크) 결과 이미지
  json/<모델>_conf<값>.json        박스 좌표(xyxy), 클래스, 점수
  masks/<모델>_conf<값>/           객체별 마스크 PNG + all_union.png (seg 모델만)
  comparison.jpg                   모델 × conf 2×2 시트
```

## 6. 도구·설정 선택 이유

- **YOLO(과제 A)**: 이미지 한 장에서 클래스·위치·점수를 바로 얻고, 결과를 이미지·JSON으로 저장하기 쉽다.
- **yolo26n / yolo26n-seg**: 같은 계열의 박스 모델과 세그멘테이션 모델이라 "마스크가 추가로 주는 정보"만 비교할 수 있다. `n`(nano)은 5~7MB라 CPU에서 한 장당 약 90~180ms(추론)로 돌아갔다.
- **imgsz=640, device=cpu**: 과제 지정 설정. 장치에 따른 차이를 배제한다.
- **conf 0.25 / 0.50**: 0.25는 ultralytics 기본값에 가까운 넓은 문턱, 0.50은 확실한 것만 남기는 엄격한 문턱이다. 둘을 비교하면 "점수가 낮은 탐지"가 무엇인지 드러난다.
- **비교NK를 버스 사진과 비교**: 버스 사진은 점수가 높아 변화가 없고, 비교NK는 작고 겹친 사람과 군함이 있어 문턱·모델 차이가 보인다.

## 7. 검토 기록·한계

원본 대조, 수정 내역, 확인 필요 항목, 미확인 범위는 [`검토기록.md`](검토기록.md)에 있다.
