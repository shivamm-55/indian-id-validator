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
- OCR
- YOLO
- Pytorch
---
# Indian ID Validator

[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model-blue)](https://huggingface.co/logasanjeev/indian-id-validator)

A robust computer vision pipeline for classifying, detecting, and extracting text from Indian identification documents, including Aadhaar, PAN Card, Passport, Voter ID, and Driving License. Powered by YOLO11 models and PaddleOCR, this project supports both front and back images for Aadhaar and Driving License.

## Overview

The **Indian ID Validator** uses deep learning to:
- **Classify** ID types (e.g., `aadhar_front`, `passport`) with the `Id_Classifier` model.
- **Detect** specific fields (e.g., Aadhaar Number, DOB, Name) using type-specific YOLO11 detection models.
- **Extract** text from detected fields via PaddleOCR with image preprocessing (upscaling, denoising, contrast enhancement).

Supported ID types:
- Aadhaar (front and back)
- PAN Card (front)
- Passport (front)
- Voter ID (front and back)
- Driving License (front and back)

## Models

### Id_Classifier
- **Model**: YOLO11l-cls
- **Classes**: `aadhar_back`, `aadhar_front`, `driving_license_back`, `driving_license_front`, `pan_card_front`, `passport`, `voter_id`
- **Metrics**:
  - Accuracy (Top-1): 0.995
  - Accuracy (Top-5): 1.0
- **Link**: [Ultralytics Hub](https://hub.ultralytics.com/models/QnJjO78MxBaRVeX2wOO4)

### Aadhaar
- **Model**: YOLO11l
- **Classes**: `Aadhaar_Number`, `Aadhaar_DOB`, `Aadhaar_Gender`, `Aadhaar_Name`, `Aadhaar_Address`
- **Metrics**:
  - mAP50: 0.795
  - mAP50-95: 0.553
  - Precision: 0.777
  - Recall: 0.774
  - Fitness: 0.577
- **Link**: [Kaggle Notebook](https://www.kaggle.com/code/ravindranlogasanjeev/aadhaar)

### Driving_License
- **Model**: YOLO11l
- **Classes**: `Address`, `Blood Group`, `DL No`, `DOB`, `Name`, `Relation With`, `RTO`, `State`, `Vehicle Type`
- **Metrics**:
  - mAP50: 0.690
  - mAP50-95: 0.524
  - Precision: 0.752
  - Recall: 0.669
- **Link**: [Ultralytics Hub](https://hub.ultralytics.com/models/eaHzQ79umKwJkic9DXbm)

### Pan_Card
- **Model**: YOLO11l
- **Classes**: `PAN`, `Name`, `Father's Name`, `DOB`, `Pan Card`
- **Metrics**:
  - mAP50: 0.924
  - mAP50-95: 0.686
  - Precision: 0.902
  - Recall: 0.901
- **Link**: [Ultralytics Hub](https://hub.ultralytics.com/models/Yj4aJ34fK02MkrHFSXq0)

### Passport
- **Model**: YOLO11l
- **Classes**: `Address`, `Code`, `DOB`, `DOI`, `EXP`, `Gender`, `MRZ1`, `MRZ2`, `Name`, `Nationality`, `Nation`, `POI`
- **Metrics**:
  - mAP50: 0.987
  - mAP50-95: 0.851
  - Precision: 0.972
  - Recall: 0.967
- **Link**: [Ultralytics Hub](https://hub.ultralytics.com/models/ELaiHGZ0bbr4JwsvSZ7z)

### Voter_Id
- **Model**: YOLO11l
- **Classes**: `Address`, `Age`, `DOB`, `Card Voter ID 1 Back`, `Card Voter ID 2 Front`, `Card Voter ID 2 Back`, `Card Voter ID 1 Front`, `Date of Issue`, `Election`, `Father`, `Gender`, `Name`, `Point`, `Portrait`, `Symbol`, `Voter ID`
- **Metrics**:
  - mAP50: 0.917
  - mAP50-95: 0.772
  - Precision: 0.922
  - Recall: 0.873
- **Link**: [Ultralytics Hub](https://hub.ultralytics.com/models/jAz7y1UQAfr2oBlwLGDp)

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://huggingface.co/logasanjeev/indian-id-validator
   cd indian-id-validator
   ```

2. **Install Dependencies**:
   Ensure Python 3.8+ is installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` includes `ultralytics`, `paddleocr`, `paddlepaddle`, `numpy==1.24.4`, `pandas==2.2.2`, and others.

3. **Download Models**:
   Models are downloaded automatically via `inference.py` from the Hugging Face repository. Ensure `config.json` is in the root directory.

## Usage

### Python API

#### Classification Only
Use `Id_Classifier` to identify the ID type:
```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO("models/Id_Classifier.pt")

# Load image
image = cv2.imread("samples/aadhaar_front.jpg")

# Classify
results = model(image)

# Print predicted class and confidence
for result in results:
    predicted_class = result.names[result.probs.top1]
    confidence = result.probs.top1conf.item()
    print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")
```
**Output**:
```
Predicted Class: aadhar_front, Confidence: 1.00
```

#### End-to-End Processing
Use `inference.py` for classification, detection, and OCR:
```python
from inference import process_id

# Process an Aadhaar back image
result = process_id(
    image_path="samples/aadhaar_back.jpg",
    save_json=True,
    output_json="detected_aadhaar_back.json",
    verbose=True
)

# Print results
import json
print(json.dumps(result, indent=2))
```
**Output**:
```json
{
  "Aadhaar": "996269466937",
  "Address": "S/O Gocala Shinde Jay Bnavani Rahiwasi Seva Sangh ..."
}
```

#### Processing a Passport with Visualizations
Process a passport image to classify, detect fields, and extract text, with visualizations enabled:
```python
from inference import process_id

# Process a passport image with verbose output
result = process_id(
    image_path="samples/passport_front.jpg",
    save_json=True,
    output_json="detected_passport.json",
    verbose=True
)

# Print results
import json
print("\nPassport Results:")
print(json.dumps(result, indent=4))
```

**Visualizations**:
The `verbose=True` flag generates visualizations for the raw image, bounding boxes, and each detected field with extracted text. Below are the results for `passport_front.jpg`:

- **Raw Image**:
  ![Raw Image](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/passport_raw_image.png)

- **Output with Bounding Boxes**:
  ![Output with Bounding Boxes](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/passport_output_bboxes.png)

- **Detected Fields**:
  - **Address**:
    ![Address](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/passport_Address.png)
  - **Code**:
    ![Code](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/passport_Code.png)
  - **DOB**:
    ![DOB](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/passport_DOB.png)
  - **DOI**:
    ![DOI](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/passport_DOI.png)
  - **EXP**:
    ![EXP](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/passport_EXP.png)
  - **Gender**:
    ![Gender](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/passport_Gender.png)
  - **MRZ1**:
    ![MRZ1](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/passport_MRZ1.png)
  - **MRZ2**:
    ![MRZ2](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/passport_MRZ2.png)
  - **Name**:
    ![Name](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/passport_Name.png)
  - **Nationality**:
    ![Nationality](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/passport_Nationality.png)
  - **Nation**:
    ![Nation](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/passport_Nation.png)
  - **POI**:
    ![POI](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/passport_POI.png)

**Output**:
```
Passport Results:
{
    "Nation": "INDIAN",
    "DOB": "26/08/1996",
    "POI": "AMRITSAR",
    "DOI": "18/06/2015",
    "Code": "NO461879",
    "EXP": "17/06/2025",
    "Address": "SHER SINGH WALAFARIDKOTASPUNJAB",
    "Name": "SHAMINDERKAUR",
    "Nationality": "IND",
    "Gender": "F",
    "MRZ1": "P<INDSANDHU<<SHAMINDER<KAUR<<<<<<<<<<<<<<<<<",
    "MRZ2": "NO461879<4IND9608269F2506171<<<<<<<<<<<<<<<2"
}
```

### Terminal
Run `inference.py` via the command line:
```bash
python inference.py samples/aadhaar_front.jpg --verbose --output-json detected_aadhaar.json
```
**Options**:
- `--model`: Specify model (e.g., `Aadhaar`, `Passport`). Default: auto-detect.
- `--no-save-json`: Disable JSON output.
- `--verbose`: Show visualizations.
- `--classify-only`: Only classify ID type.

**Example Output**:
```
Detected document type: aadhar_front with confidence: 0.98
Extracted Text:
{
  "Aadhaar": "1234 5678 9012",
  "DOB": "01/01/1990",
  "Gender": "M",
  "Name": "John Doe",
  "Address": "123 Main St, City, State"
}
```

## Colab Tutorial

Try the interactive tutorial to test the model with sample images or your own:
[Open in Colab](https://colab.research.google.com/drive/1_hIvuJ9p1kx8wKTG1ThK9vV8ijiNTlPX)

## Links

- **Repository**: [Hugging Face](https://huggingface.co/logasanjeev/indian-id-validator)
- **Models**:
  - Id_Classifier: [Ultralytics](https://hub.ultralytics.com/models/QnJjO78MxBaRVeX2wOO4)
  - Aadhaar: [Kaggle](https://www.kaggle.com/code/ravindranlogasanjeev/aadhaar)
  - Pan_Card: [Ultralytics](https://hub.ultralytics.com/models/Yj4aJ34fK02MkrHFSXq0)
  - Passport: [Ultralytics](https://hub.ultralytics.com/models/ELaiHGZ0bbr4JwsvSZ7z)
  - Voter_Id: [Ultralytics](https://hub.ultralytics.com/models/jAz7y1UQAfr2oBlwLGDp)
  - Driving_License: [Ultralytics](https://hub.ultralytics.com/models/eaHzQ79umKwJkic9DXbm)
- **Tutorial**: [Colab Notebook](https://colab.research.google.com/drive/1_hIvuJ9p1kx8wKTG1ThK9vV8ijiNTlPX)
- **Inference Script**: [inference.py](https://huggingface.co/logasanjeev/indian-id-validator/blob/main/inference.py)
- **Config**: [config.json](https://huggingface.co/logasanjeev/indian-id-validator/blob/main/config.json)

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a branch: `git checkout -b feature-name`.
3. Submit a pull request with your changes.

Report issues or suggest features via the [Hugging Face Issues](https://huggingface.co/logasanjeev/indian-id-validator/discussions) page.