import json
import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from paddleocr import PaddleOCR
from pathlib import Path

def load_config(config_path="config.json"):
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found.")
    with open(config_path, 'r') as f:
        return json.load(f)

def preprocess_image(image):
    """Apply preprocessing steps to enhance OCR accuracy."""
    scale_factor = 2
    image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image = cv2.filter2D(image, -1, kernel)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(gray)
    
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

def run_ocr(cropped_image, ocr):
    """Run PaddleOCR on a cropped image and return extracted text with confidence."""
    result = ocr.ocr(cropped_image, cls=True)
    if not result or not result[0]:
        return None, 0.0
    text = result[0][0][1][0]
    confidence = result[0][0][1][1]
    return text, confidence

def visualize_yolo_output(image, boxes, class_names, save_path=None, show=False):
    """Visualize YOLO bounding boxes on the image."""
    img = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
        label = class_names[int(box.cls)]
        conf = box.conf[0].numpy()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{label}: {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    if save_path:
        cv2.imwrite(save_path, img)
    if show:
        plt.imshow(img[:, :, ::-1])
        plt.axis('off')
        plt.show()
    return img

def visualize_ocr_output(cropped_image, ocr_result, text, confidence, save_path=None, show=False):
    """Visualize OCR bounding boxes and text on the cropped image."""
    img = cropped_image.copy()
    if ocr_result and ocr_result[0]:
        for line in ocr_result[0]:
            box = line[0]
            x1, y1 = int(box[0][0]), int(box[0][1])
            x2, y2 = int(box[2][0]), int(box[2][1])
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"{text} ({confidence:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    if save_path:
        cv2.imwrite(save_path, img)
    if show:
        plt.imshow(img[:, :, ::-1])
        plt.axis('off')
        plt.show()
    return img

def process_image(image_path, config, model_choice=None, show_yolo=False, show_ocr=False, save_json=True, verbose=False):
    """Process an input image to classify document type, detect fields, and extract text."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image {image_path} not found.")
    
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image {image_path}.")
    
    ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    
    doc_type = model_choice
    if model_choice is None:
        classifier = YOLO(config["models"]["id_classifier"]["path"])
        results = classifier(image, verbose=verbose)
        top_class_idx = results[0].probs.top1
        doc_type = config["models"]["id_classifier"]["classes"][str(top_class_idx)]
        if verbose:
            print(f"Classified document as: {doc_type} (confidence: {results[0].probs.top1conf:.2f})")
    
    if doc_type not in config["doc_type_to_model"]:
        raise ValueError(f"Document type {doc_type} not supported.")
    model_name = config["doc_type_to_model"][doc_type]
    if model_name not in config["models"]:
        raise ValueError(f"Model {model_name} not found in config.")
    
    detector = YOLO(config["models"][model_name]["path"])
    class_names = config["models"][model_name]["classes"]
    results = detector(image, verbose=verbose)
    
    output = {}
    
    for i, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = box.xyxy[0].numpy().astype(int)
        label = class_names[int(box.cls)]
        conf = box.conf[0].numpy()
        
        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            continue
        
        preprocessed = preprocess_image(cropped)
        
        text, ocr_conf = run_ocr(preprocessed, ocr)
        if text:
            output[label] = {"text": text, "yolo_conf": float(conf), "ocr_conf": float(ocr_conf)}
            if verbose:
                print(f"Field: {label}, Text: {text}, YOLO Conf: {conf:.2f}, OCR Conf: {ocr_conf:.2f}")
        
        if show_ocr or (save_json and show_ocr):
            ocr_result = ocr.ocr(preprocessed, cls=True)
            save_path = f"ocr_output_{label}_{i}.jpg" if save_json else None
            visualize_ocr_output(preprocessed, ocr_result, text, ocr_conf, save_path=save_path, show=show_ocr)
    
    if show_yolo or (save_json and show_yolo):
        save_path = "yolo_output.jpg" if save_json else None
        visualize_yolo_output(image, results[0].boxes, class_names, save_path=save_path, show=show_yolo)
    
    if save_json:
        output_path = "detected_text.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        if verbose:
            print(f"Saved results to {output_path}")
    
    return output

def main():
    """Command-line interface for inference."""
    parser = argparse.ArgumentParser(description="Indian ID Validator Inference Script")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default=None, choices=["aadhaar", "pan_card", "passport", "voter_id", "driving_license"],
                        help="Specify detection model (default: auto via id_classifier)")
    parser.add_argument("--show-yolo", action="store_true", help="Display/save YOLO bounding box image")
    parser.add_argument("--show-ocr", action="store_true", help="Display/save OCR results for each field")
    parser.add_argument("--no-save-json", action="store_true", help="Disable saving detected_text.json")
    parser.add_argument("--verbose", action="store_true", help="Print detailed inference results")
    args = parser.parse_args()
    
    config = load_config()
    try:
        output = process_image(
            image_path=args.image,
            config=config,
            model_choice=args.model,
            show_yolo=args.show_yolo,
            show_ocr=args.show_ocr,
            save_json=not args.no_save_json,
            verbose=args.verbose
        )
        if not args.verbose:
            print("Detected Fields:")
            for label, data in output.items():
                print(f"{label}: {data['text']} (YOLO Conf: {data['yolo_conf']:.2f}, OCR Conf: {data['ocr_conf']:.2f})")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()