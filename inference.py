import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from paddleocr import PaddleOCR
from huggingface_hub import hf_hub_download
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Suppress ultralytics verbose output
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Load configuration
def load_config(config_path="config.json"):
    if not os.path.exists(config_path):
        config_path = hf_hub_download(repo_id="logasanjeev/indian-id-validator", filename="config.json")
    with open(config_path, "r") as f:
        return json.load(f)

CONFIG = load_config()

# Initialize PaddleOCR
OCR = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

# Preprocessing functions
def upscale_image(image, scale=2):
    """Upscales the image to improve OCR accuracy."""
    return cv2.resize(image, (image.shape[1] * scale, image.shape[0] * scale), interpolation=cv2.INTER_CUBIC)

def unblur_image(image):
    """Sharpens the image to reduce blurriness."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def denoise_image(image):
    """Removes noise using Non-Local Means Denoising."""
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def enhance_contrast(image):
    """Enhances contrast using CLAHE."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def preprocess_image(image):
    """Applies all preprocessing steps."""
    if isinstance(image, str):
        image = cv2.imread(image)
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("Invalid image input. Provide a valid file path or numpy array.")
    image = upscale_image(image, scale=2)
    image = unblur_image(image)
    image = denoise_image(image)
    image = enhance_contrast(image)
    return image

# Core inference function
def process_id(image_path, model_name=None, save_json=True, output_json="detected_text.json", verbose=False, classify_only=False):
    """
    Process an ID image to classify document type, detect fields, and extract text.
    
    Args:
        image_path (str): Path to the input image.
        model_name (str, optional): Specific model to use. If None, uses Id_Classifier.
        save_json (bool): Save extracted text to JSON file.
        output_json (str): Path to save JSON output.
        verbose (bool): Display visualizations.
        classify_only (bool): If True, only classify document type and return result.
    
    Returns:
        dict: Extracted text for each detected field, or {} for unmapped document types or classify_only.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Download and load model
    def load_model(model_key):
        model_path = CONFIG["models"][model_key]["path"]
        if not os.path.exists(model_path):
            model_path = hf_hub_download(repo_id="logasanjeev/indian-id-validator", filename=model_path)
        return YOLO(model_path)

    # Classify document type if model_name is not specified
    if model_name is None:
        classifier = load_model("Id_Classifier")
        results = classifier(image)
        doc_type = results[0].names[results[0].probs.top1]
        confidence = results[0].probs.top1conf.item()
        print(f"Detected document type: {doc_type} with confidence: {confidence:.2f}")
        logger.info(f"Detected document type: {doc_type}, confidence: {confidence:.2f}")
        if classify_only:
            return {"doc_type": doc_type, "confidence": confidence}
        model_name = CONFIG["doc_type_to_model"].get(doc_type, None)
        if model_name is None:
            logger.warning(f"No detection model mapped for document type: {doc_type}. Returning empty result.")
            if save_json:
                with open(output_json, "w") as f:
                    json.dump({}, f, indent=4)
            return {}

    # Load detection model
    if model_name not in CONFIG["models"]:
        raise ValueError(f"Invalid model name: {model_name}")
    model = load_model(model_name)
    class_names = CONFIG["models"][model_name]["classes"]
    logger.info(f"Loaded model: {model_name} with classes: {class_names}")

    # Run inference
    results = model(image_path)
    filtered_boxes = {}
    output_image = results[0].orig_img.copy()
    original_image = cv2.imread(image_path)
    h, w, _ = output_image.shape

    # Filter highest confidence box for each class
    for result in results:
        if not result.boxes:
            logger.warning("No boxes detected in the image.")
            continue
        for box in result.boxes:
            try:
                cls = int(box.cls[0].item())
                if cls >= len(class_names):
                    logger.warning(f"Invalid class index {cls} for model {model_name}. Skipping box.")
                    continue
                conf = box.conf[0].item()
                xyxy = box.xyxy[0].tolist()
                class_name = class_names[cls]
                logger.info(f"Detected box for class index: {cls}, class name: {class_name}, confidence: {conf:.2f}, coords: {xyxy}")
                if cls not in filtered_boxes or conf > filtered_boxes[cls]["conf"]:
                    filtered_boxes[cls] = {"conf": conf, "xyxy": xyxy, "class_name": class_name}
            except IndexError as e:
                logger.error(f"Error processing box: {e}, box data: {box}")
                continue

    # Extract text and visualize
    detected_text = {}
    processed_images = []
    for cls, data in filtered_boxes.items():
        try:
            x_min, y_min, x_max, y_max = map(int, data["xyxy"])
            class_name = data["class_name"]
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)
            logger.info(f"Processing class: {class_name} at coordinates: ({x_min}, {y_min}, {x_max}, {y_max})")

            # Crop region
            region_img = original_image[y_min:y_max, x_min:x_max]
            if region_img.size == 0:
                logger.warning(f"Empty region for class: {class_name}. Skipping.")
                continue
            region_img = preprocess_image(region_img)
            region_h, region_w = region_img.shape[:2]

            # Create black canvas and center the cropped region
            black_canvas = np.ones((h, w, 3), dtype=np.uint8)
            center_x, center_y = w // 2, h // 2
            top_left_x = max(0, min(w - region_w, center_x - region_w // 2))
            top_left_y = max(0, min(h - region_h, center_y - region_h // 2))
            region_w = min(region_w, w - top_left_x)
            region_h = min(region_h, h - top_left_y)
            region_img = cv2.resize(region_img, (region_w, region_h))
            black_canvas[top_left_y:top_left_y+region_h, top_left_x:top_left_x+region_w] = region_img

            # Perform OCR
            ocr_result = OCR.ocr(black_canvas, cls=True)
            if ocr_result is None or not ocr_result:
                logger.warning(f"No OCR result for class: {class_name}. Skipping.")
                detected_text[class_name] = "No text detected"
                continue
            extracted_text = []
            for line in ocr_result:
                if line is None:
                    continue
                for word_info in line:
                    if word_info is None or len(word_info) < 2 or not word_info[1]:
                        continue
                    extracted_text.append(word_info[1][0])
            extracted_text = " ".join(extracted_text) if extracted_text else "No text detected"
            logger.info(f"Extracted text for {class_name}: {extracted_text}")
            detected_text[class_name] = extracted_text

            # Draw OCR bounding boxes
            for line in ocr_result:
                if line is None:
                    continue
                for word_info in line:
                    if word_info is None or len(word_info) < 1:
                        continue
                    try:
                        box = word_info[0]
                        x1, y1 = int(box[0][0]), int(box[0][1])
                        x2, y2 = int(box[2][0]), int(box[2][1])
                        cv2.rectangle(black_canvas, (x1, y1), (x2, y2), (0, 255, 0), 5)
                    except (IndexError, TypeError) as e:
                        logger.error(f"Error drawing OCR box for class {class_name}: {e}")
                        continue

            # Save processed image
            processed_images.append((class_name, black_canvas, extracted_text))

            # Draw original bounding box
            cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(output_image, class_name, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        except Exception as e:
            logger.error(f"Error processing class {class_name}: {e}")
            continue

    # Save JSON
    if save_json:
        with open(output_json, "w") as f:
            json.dump(detected_text, f, indent=4)

    # Visualize
    if verbose:
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Raw Image")
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Output Image with Bounding Boxes")
        plt.show()

        for class_name, cropped_image, text in processed_images:
            plt.figure(figsize=(10, 10))
            plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
            plt.axis("off")
            plt.title(f"{class_name} - Extracted: {text}")
            plt.show()

    return detected_text

# Model-specific functions
def aadhaar(image_path, save_json=True, output_json="detected_text.json", verbose=False):
    """Process an Aadhaar card image."""
    return process_id(image_path, model_name="Aadhaar", save_json=save_json, output_json=output_json, verbose=verbose)

def pan_card(image_path, save_json=True, output_json="detected_text.json", verbose=False):
    """Process a PAN card image."""
    return process_id(image_path, model_name="Pan_Card", save_json=save_json, output_json=output_json, verbose=verbose)

def passport(image_path, save_json=True, output_json="detected_text.json", verbose=False):
    """Process a passport image."""
    return process_id(image_path, model_name="Passport", save_json=save_json, output_json=output_json, verbose=verbose)

def voter_id(image_path, save_json=True, output_json="detected_text.json", verbose=False):
    """Process a voter ID image."""
    return process_id(image_path, model_name="Voter_Id", save_json=save_json, output_json=output_json, verbose=verbose)

def driving_license(image_path, save_json=True, output_json="detected_text.json", verbose=False):
    """Process a driving license image."""
    return process_id(image_path, model_name="Driving_License", save_json=save_json, output_json=output_json, verbose=verbose)

# Command-line interface
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Indian ID Validator: Classify and extract fields from ID images.")
    parser.add_argument("image_path", help="Path to the input ID image")
    parser.add_argument("--model", default=None, choices=["Aadhaar", "Pan_Card", "Passport", "Voter_Id", "Driving_License"],
                        help="Specific model to use (default: auto-detect with Id_Classifier)")
    parser.add_argument("--no-save-json", action="store_false", dest="save_json", help="Disable saving to JSON")
    parser.add_argument("--output-json", default="detected_text.json", help="Path to save JSON output")
    parser.add_argument("--verbose", action="store_true", help="Display visualizations")
    parser.add_argument("--classify-only", action="store_true", dest="classify_only", help="Only classify document type")
    args = parser.parse_args()

    result = process_id(args.image_path, args.model, args.save_json, args.output_json, args.verbose, args.classify_only)
    print("Extracted Text:")
    print(json.dumps(result, indent=4))