---
license: mit
language:
- en
metrics:
- accuracy
- recall
- precision
- mean_iou
base_model:
- Ultralytics/YOLO11
pipeline_tag: image-to-text
tags:
- ocr
- yolo
- pytorch
- paddlepaddle
- computer-vision
- image-classification
- object-detection
- indian-id
- document-processing
- ultralytics
model-index:
- name: Id_Classifier
  results:
  - task:
      type: image-classification
    dataset:
      name: custom-indian-id-dataset
      type: custom-indian-id-dataset
    metrics:
    - name: Accuracy (Top-1)
      type: accuracy_top1
      value: 0.995
    - name: Accuracy (Top-5)
      type: accuracy_top5
      value: 1
    source:
      name: Ultralytics Hub
      url: https://hub.ultralytics.com/models/QnJjO78MxBaRVeX2wOO4
- name: Aadhaar
  results:
  - task:
      type: object-detection
    dataset:
      name: custom-indian-id-dataset
      type: custom-indian-id-dataset
    metrics:
    - name: mAP50
      type: mAP50
      value: 0.795
    - name: mAP50-95
      type: mAP50-95
      value: 0.553
    - name: Precision
      type: precision
      value: 0.777
    - name: Recall
      type: recall
      value: 0.774
    - name: Fitness
      type: fitness
      value: 0.577
    source:
      name: Kaggle Notebook
      url: https://www.kaggle.com/code/ravindranlogasanjeev/aadhaar
- name: Driving_License
  results:
  - task:
      type: object-detection
    dataset:
      name: custom-indian-id-dataset
      type: custom-indian-id-dataset
    metrics:
    - name: mAP50
      type: mAP50
      value: 0.69
    - name: mAP50-95
      type: mAP50-95
      value: 0.524
    - name: Precision
      type: precision
      value: 0.752
    - name: Recall
      type: recall
      value: 0.669
    source:
      name: Ultralytics Hub
      url: https://hub.ultralytics.com/models/eaHzQ79umKwJkic9DXbm
- name: Pan_Card
  results:
  - task:
      type: object-detection
    dataset:
      name: custom-indian-id-dataset
      type: custom-indian-id-dataset
    metrics:
    - name: mAP50
      type: mAP50
      value: 0.924
    - name: mAP50-95
      type: mAP50-95
      value: 0.686
    - name: Precision
      type: precision
      value: 0.902
    - name: Recall
      type: recall
      value: 0.901
    source:
      name: Ultralytics Hub
      url: https://hub.ultralytics.com/models/Yj4aJ34fK02MkrHFSXq0
- name: Passport
  results:
  - task:
      type: object-detection
    dataset:
      name: custom-indian-id-dataset
      type: custom-indian-id-dataset
    metrics:
    - name: mAP50
      type: mAP50
      value: 0.987
    - name: mAP50-95
      type: mAP50-95
      value: 0.851
    - name: Precision
      type: precision
      value: 0.972
    - name: Recall
      type: recall
      value: 0.967
    source:
      name: Ultralytics Hub
      url: https://hub.ultralytics.com/models/ELaiHGZ0bbr4JwsvSZ7z
- name: Voter_Id
  results:
  - task:
      type: object-detection
    dataset:
      name: custom-indian-id-dataset
      type: custom-indian-id-dataset
    metrics:
    - name: mAP50
      type: mAP50
      value: 0.917
    - name: mAP50-95
      type: mAP50-95
      value: 0.772
    - name: Precision
      type: precision
      value: 0.922
    - name: Recall
      type: recall
      value: 0.873
    source:
      name: Ultralytics Hub
      url: https://hub.ultralytics.com/models/jAz7y1UQAfr2oBlwLGDp
---
# Indian ID Validator

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/shivamm-55/indian-id-validator)

A robust computer vision and OCR pipeline for classifying, validating, and extracting data from Indian identification documents. Powered by **YOLO11** detection/classification models and **PaddleOCR**, this project supports **Aadhaar, PAN Card, Passport, Voter ID, and Driving License**.

It includes a multi-step validation engine that verifies document type and side, performs layout-based side resolution fallbacks, and executes secure cross-side matching for dual-side uploads.

---

## Key Features

1. **Automatic Document Classification**: Identifies the document type and side (`aadhar_front`, `driving_license_back`, `passport`, etc.) using a custom `Id_Classifier` model.
2. **Targeted Field Detection**: Localizes regions of interest (e.g. Aadhaar Number, DOB, Name, Address) using card-specific YOLO11 models.
3. **Advanced Image Preprocessing**: Enhances cropped text regions (via upscaling, sharpening, denoising, and CLAHE contrast adjustments) to maximize OCR extraction accuracy.
4. **Latency-Optimized Cross-Linking**: Runs OCR *only* on the specific unique identifier bounding box during dual-side validation, bringing CPU response times down from 50s to **under 5 seconds** (a 90% latency reduction).
5. **Robust Document Side Fallback**: Intelligently resolves and corrects card side classification errors (for Voter ID and Driving Licenses) by detecting layout-specific fields (e.g., matching `Address` to the back cover).
6. **Masked ID Validation**: Employs strict suffix-matching rules to securely verify masked Aadhaar front cards (e.g., verifying that the 4-digit front suffix matches the end of the full 12-digit back number).
7. **REST API**: Packaged with a lightweight **FastAPI** web server to easily process and validate documents remotely.

---

## Supported ID Types & Fields

| ID Type | Layouts | Bounding Box Classes Detected |
|---|---|---|
| **Aadhaar** | Front & Back | `Aadhaar` (Number), `DOB`, `Gender`, `Name`, `Address` |
| **PAN Card** | Front | `PAN`, `Name`, `Father's Name`, `DOB`, `Pan Card` (whole card) |
| **Driving License** | Front & Back | `Address`, `Blood Group`, `DL No`, `DOB`, `Name`, `Relation With`, `RTO`, `State`, `Vehicle Type` |
| **Passport** | Details Page | `Address`, `Code`, `DOB`, `DOI`, `EXP`, `Gender`, `MRZ1`, `MRZ2`, `Name`, `Nationality`, `Nation`, `POI` |
| **Voter ID** | Front & Back | `Address`, `Age`, `DOB`, `Date of Issue`, `Election`, `Father`, `Gender`, `Name`, `Portrait`, `Symbol`, `Voter ID` |

---

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/shivamm-55/indian-id-validator
cd indian-id-validator
```

### 2. Install Dependencies
Ensure Python 3.8+ is installed, then run:
```bash
pip install -r requirements.txt
pip install fastapi uvicorn python-multipart
```

### 3. Model Downloads
Models are downloaded automatically from Hugging Face Hub during script execution. If you have Git LFS placeholder files (133 bytes), the code automatically detects and overwrites them with the full weights.

---

## Usage

### 1. FastAPI REST API (Recommended)
You can start the FastAPI web service to receive validations over HTTP:
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

#### Test curl Command (Dual-Side Aadhaar Validation)
```bash
curl -X POST \
  -F "front_image=@samples/aadhaar_front.jpg" \
  -F "back_image=@samples/aadhaar_back.jpg" \
  -F "expected_type=Aadhaar" \
  http://127.0.0.1:8000/validate
```

---

### 2. Python API
Integrate the validation engine directly into your code:

```python
from validation import validate_document

# Validate dual-side uploads
report = validate_document(
    front_image_path="samples/aadhaar_front.jpg",
    back_image_path="samples/aadhaar_back.jpg",
    expected_type="Aadhaar",
    confidence_threshold=0.75
)

print(report["is_valid"])  # Returns True/False
print(report["status"])    # Returns "success" or "mismatch"
```

---

### 3. CLI Interface

#### Full End-to-End Extraction:
```bash
python inference.py samples/aadhaar_front.jpg --verbose --output-json output.json
```

#### Document Validation CLI:
```bash
python validation.py --front samples/aadhaar_front.jpg --back samples/aadhaar_back.jpg --expected-type Aadhaar
```

---

## Colab Tutorial
Try out the interactive tutorial notebook to test the pipeline online:
[Open in Colab](https://colab.research.google.com/drive/1_hIvuJ9p1kx8wKTG1ThK9vV8ijiNTlPX)

---

## License
MIT License