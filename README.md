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

[![Hugging Face Model](https://img.shields.io/badge/Hugging%20Face-Model-blue)](https://huggingface.co/logasanjeev/indian-id-validator)

A robust computer vision pipeline for classifying, detecting, and extracting text from Indian identification documents, including Aadhaar, PAN Card, Passport, Voter ID, and Driving License. Powered by YOLO11 models and PaddleOCR, this project supports both front and back images for Aadhaar and Driving License.

## Overview

The **Indian ID Validator** uses deep learning to:
- **Classify** ID types (e.g., `aadhar_front`, `passport`) with the `Id_Classifier` model.
- **Detect** specific fields (e.g., Aadhaar Number, DOB, Name) using type-specific YOLO11 detection models.
- **Extract** text from detected fields via PaddleOCR with image preprocessing (upscaling, denoising, contrast enhancement).

**Supported ID Types**:
- Aadhaar (front and back)
- PAN Card (front)
- Passport (front)
- Voter ID (front and back)
- Driving License (front and back)

## Models

The pipeline consists of the following models, each designed for specific tasks in the ID validation process. Models can be downloaded from their respective Ultralytics Hub links in various formats such as PyTorch, ONNX, TensorRT, and more for deployment in different environments.

| Model Name       | Type        | Description                                                                                   | Link                                      |
|------------------|-------------|-----------------------------------------------------------------------------------------------|-------------------------------------------|
| Id_Classifier    | YOLO11l-cls | Classifies the type of Indian ID document (e.g., Aadhaar, Passport).                          | [Ultralytics Hub](https://hub.ultralytics.com/models/QnJjO78MxBaRVeX2wOO4) |
| Aadhaar          | YOLO11l     | Detects fields on Aadhaar cards (front and back), such as Aadhaar Number, DOB, and Address.   | [Kaggle Notebook](https://www.kaggle.com/code/ravindranlogasanjeev/aadhaar) |
| Driving_License  | YOLO11l     | Detects fields on Driving Licenses (front and back), including DL No, DOB, and Vehicle Type.  | [Ultralytics Hub](https://hub.ultralytics.com/models/eaHzQ79umKwJkic9DXbm) |
| Pan_Card         | YOLO11l     | Detects fields on PAN Cards, such as PAN Number, Name, and DOB.                               | [Ultralytics Hub](https://hub.ultralytics.com/models/Yj4aJ34fK02MkrHFSXq0) |
| Passport         | YOLO11l     | Detects fields on Passports, including MRZ lines, DOB, and Nationality.                       | [Ultralytics Hub](https://hub.ultralytics.com/models/ELaiHGZ0bbr4JwsvSZ7z) |
| Voter_Id         | YOLO11l     | Detects fields on Voter ID cards (front and back), such as Voter ID, Name, and Address.       | [Ultralytics Hub](https://hub.ultralytics.com/models/jAz7y1UQAfr2oBlwLGDp) |

## Model Details

Below is a detailed breakdown of each model, including the classes they detect and their evaluation metrics on a custom Indian ID dataset.

| Model Name       | Task                | Classes                                                                                   | Metrics                                                                                   |
|------------------|---------------------|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| **Id_Classifier**| Image Classification| `aadhar_back`, `aadhar_front`, `driving_license_back`, `driving_license_front`, `pan_card_front`, `passport`, `voter_id` | Accuracy (Top-1): 0.995, Accuracy (Top-5): 1.0                                           |
| **Aadhaar**      | Object Detection    | `Aadhaar_Number`, `Aadhaar_DOB`, `Aadhaar_Gender`, `Aadhaar_Name`, `Aadhaar_Address`     | mAP50: 0.795, mAP50-95: 0.553, Precision: 0.777, Recall: 0.774, Fitness: 0.577          |
| **Driving_License**| Object Detection  | `Address`, `Blood Group`, `DL No`, `DOB`, `Name`, `Relation With`, `RTO`, `State`, `Vehicle Type` | mAP50: 0.690, mAP50-95: 0.524, Precision: 0.752, Recall: 0.669                           |
| **Pan_Card**     | Object Detection    | `PAN`, `Name`, `Father's Name`, `DOB`, `Pan Card`                                        | mAP50: 0.924, mAP50-95: 0.686, Precision: 0.902, Recall: 0.901                           |
| **Passport**     | Object Detection    | `Address`, `Code`, `DOB`, `DOI`, `EXP`, `Gender`, `MRZ1`, `MRZ2`, `Name`, `Nationality`, `Nation`, `POI` | mAP50: 0.987, mAP50-95: 0.851, Precision: 0.972, Recall: 0.967                           |
| **Voter_Id**     | Object Detection    | `Address`, `Age`, `DOB`, `Card Voter ID 1 Back`, `Card Voter ID 2 Front`, `Card Voter ID 2 Back`, `Card Voter ID 1 Front`, `Date of Issue`, `Election`, `Father`, `Gender`, `Name`, `Point`, `Portrait`, `Symbol`, `Voter ID` | mAP50: 0.917, mAP50-95: 0.772, Precision: 0.922, Recall: 0.873                           |

For additional details, refer to the `model-index` section in the YAML metadata at the top of this README.

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
   Models are downloaded automatically via `inference.py` from the Hugging Face repository. Ensure `config.json` is in the root directory. Alternatively, use the Ultralytics Hub links above to download models in formats like PyTorch, ONNX, etc.

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

| **Type**                     | **Image**                                                                                     |
|------------------------------|-----------------------------------------------------------------------------------------------|
| **Raw Image**                | ![Raw Image](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/Picture1.jpeg) |
| **Output with Bounding Boxes** | ![Output with Bounding Boxes](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/Picture2.jpeg) |

**Detected Fields**:

| **Field**      | **Image**                                                                                     |
|----------------|-----------------------------------------------------------------------------------------------|
| **Address**    | ![Address](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/Picture9.png) |
| **Code**       | ![Code](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/Picture7.png) |
| **DOB**        | ![DOB](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/Picture4.png) |
| **DOI**        | ![DOI](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/Picture6.png) |
| **EXP**        | ![EXP](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/Picture8.png) |
| **Gender**     | ![Gender](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/Picture12.png) |
| **MRZ1**       | ![MRZ1](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/Picture13.png) |
| **MRZ2**       | ![MRZ2](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/Picture14.png) |
| **Name**       | ![Name](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/Picture10.png) |
| **Nationality**| ![Nationality](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/Picture11.png) |
| **Nation**     | ![Nation](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/Picture3.png) |
| **POI**        | ![POI](https://huggingface.co/logasanjeev/indian-id-validator/raw/main/results/Picture5.png) |

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

## License

MIT License