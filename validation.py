import os
import cv2
import json
import logging
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Import required components from the main pipeline
from inference import CONFIG, process_id

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def normalize_document_type(expected_type):
    """
    Normalizes expected document type string into the canonical model names
    used in config.json (e.g. Aadhaar, Pan_Card, Driving_License, Passport, Voter_Id).
    """
    if not expected_type or not isinstance(expected_type, str):
        return None
        
    # Convert to lowercase and strip all spaces, underscores, and hyphens
    clean = expected_type.lower().replace(" ", "").replace("_", "").replace("-", "")
    
    # Remove common suffix noise words
    for word in ["card", "license", "id"]:
        clean = clean.replace(word, "")
        
    mapping = {
        "aadhar": "Aadhaar",
        "aadhaar": "Aadhaar",
        "pan": "Pan_Card",
        "passport": "Passport",
        "voter": "Voter_Id",
        "dl": "Driving_License",
        "driving": "Driving_License"
    }
    return mapping.get(clean, expected_type)

def load_yolo_model(model_key):
    """Loads a YOLO model based on the key in config.json, downloading it if not present."""
    if model_key not in CONFIG["models"]:
        raise ValueError(f"Model key '{model_key}' not found in configuration.")
    model_path = CONFIG["models"][model_key]["path"]
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000:
        logger.info(f"Model {model_key} not found or is placeholder at {model_path}. Downloading from Hugging Face Hub...")
        model_path = hf_hub_download(repo_id="logasanjeev/indian-id-validator", filename=model_path)
    return YOLO(model_path)

def determine_voter_id_side(image_path):
    """
    Determines whether a Voter ID image is the front or back side.
    Since the Id_Classifier outputs a generic 'voter_id' class, this function
    uses the custom Voter_Id detection model to inspect detected features.
    """
    try:
        voter_model = load_yolo_model("Voter_Id")
        results = voter_model(image_path)
        
        front_score = 0.0
        back_score = 0.0
        
        # Front vs Back indicator classes as defined in Voter ID detection model
        front_classes = {"Card Voter ID 1 Front", "Card Voter ID 2 Front", "Portrait", "Symbol", "Name", "Father", "Age", "DOB", "Election"}
        back_classes = {"Card Voter ID 1 Back", "Card Voter ID 2 Back", "Address", "Date of Issue"}
        
        class_names = CONFIG["models"]["Voter_Id"]["classes"]
        
        for result in results:
            if not result.boxes:
                continue
            for box in result.boxes:
                cls_idx = int(box.cls[0].item())
                conf = box.conf[0].item()
                if cls_idx < len(class_names):
                    class_name = class_names[cls_idx]
                    if class_name in front_classes:
                        front_score += conf
                    elif class_name in back_classes:
                        back_score += conf
        
        logger.info(f"Voter ID side determination scores -> Front: {front_score:.2f}, Back: {back_score:.2f}")
        
        # If score is too low or equal, we return None (unable to determine side confidently)
        if front_score > back_score and front_score > 0.3:
            return "front"
        elif back_score > front_score and back_score > 0.3:
            return "back"
    except Exception as e:
        logger.error(f"Error determining Voter ID side: {e}")
    
    return None

def determine_driving_license_side(image_path):
    """
    Determines whether a Driving License image is the front or back side by checking detected fields.
    Useful for correcting classification errors when the classifier confuses front and back.
    """
    try:
        dl_model = load_yolo_model("Driving_License")
        results = dl_model(image_path)
        
        front_score = 0.0
        back_score = 0.0
        
        # DL field classes
        # Front indicators: "DL No", "Name", "DOB", "Relation With"
        # Back indicators: "Address", "Vehicle Type", "RTO", "State"
        front_classes = {"DL No", "Name", "DOB", "Relation With", "Blood Group"}
        back_classes = {"Address", "Vehicle Type", "RTO", "State"}
        
        class_names = CONFIG["models"]["Driving_License"]["classes"]
        
        for result in results:
            if not result.boxes:
                continue
            for box in result.boxes:
                cls_idx = int(box.cls[0].item())
                conf = box.conf[0].item()
                if cls_idx < len(class_names):
                    class_name = class_names[cls_idx]
                    if class_name in front_classes:
                        front_score += conf
                    elif class_name in back_classes:
                        back_score += conf
        
        logger.info(f"DL side determination scores -> Front: {front_score:.2f}, Back: {back_score:.2f}")
        
        if front_score > back_score and front_score > 0.3:
            return "front"
        elif back_score > front_score and back_score > 0.3:
            return "back"
    except Exception as e:
        logger.error(f"Error determining Driving License side: {e}")
    
    return None

def extract_linking_identifier(image_path, doc_type):
    """
    Optimized function to extract ONLY the unique ID number field,
    avoiding OCR on other fields (Name, Address, DOB, etc.) to save time.
    """
    try:
        import numpy as np
        from inference import preprocess_image, OCR
        
        # 1. Map doc_type to the specific class we want to detect
        doc_to_class = {
            "Aadhaar": "Aadhaar",
            "Driving_License": "DL No",
            "Voter_Id": "Voter ID",
            "Pan_Card": "PAN"
        }
        
        target_class = doc_to_class.get(doc_type)
        # For Passport, we need MRZ1, MRZ2, or Code (so multiple classes)
        target_classes = {target_class} if target_class else {"MRZ1", "MRZ2", "Code"}
        
        # 2. Load the detection model
        model = load_yolo_model(doc_type)
        class_names = CONFIG["models"][doc_type]["classes"]
        
        # 3. Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        h, w, _ = image.shape
        
        # 4. Run detection
        results = model(image_path)
        
        # Find the highest confidence box for target classes
        best_box = None
        best_conf = 0.0
        best_class = None
        
        for result in results:
            if not result.boxes:
                continue
            for box in result.boxes:
                cls_idx = int(box.cls[0].item())
                if cls_idx >= len(class_names):
                    continue
                class_name = class_names[cls_idx]
                if class_name in target_classes:
                    conf = box.conf[0].item()
                    if conf > best_conf:
                        best_conf = conf
                        best_box = box.xyxy[0].tolist()
                        best_class = class_name
                        
        if best_box is None:
            logger.warning(f"Target identifier class {target_classes} not detected in {image_path}.")
            return None
            
        # 5. Crop the single best bounding box
        x_min, y_min, x_max, y_max = map(int, best_box)
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)
        
        region_img = image[y_min:y_max, x_min:x_max]
        if region_img.size == 0:
            return None
            
        # 6. Preprocess ONLY this crop
        region_img = preprocess_image(region_img)
        region_h, region_w = region_img.shape[:2]
        
        # Center in a black canvas to aid OCR (as in inference.py)
        black_canvas = np.ones((h, w, 3), dtype=np.uint8)
        center_x, center_y = w // 2, h // 2
        top_left_x = max(0, min(w - region_w, center_x - region_w // 2))
        top_left_y = max(0, min(h - region_h, center_y - region_h // 2))
        region_w = min(region_w, w - top_left_x)
        region_h = min(region_h, h - top_left_y)
        region_img = cv2.resize(region_img, (region_w, region_h))
        black_canvas[top_left_y:top_left_y+region_h, top_left_x:top_left_x+region_w] = region_img
        
        # 7. Run OCR
        ocr_result = OCR.ocr(black_canvas)
        if not ocr_result:
            return None
            
        extracted_text_list = []
        for item in ocr_result:
            if isinstance(item, dict):
                rec_texts = item.get("rec_texts", [])
                extracted_text_list.extend(rec_texts)
            elif isinstance(item, list):
                for word_info in item:
                    if isinstance(word_info, list) and len(word_info) >= 2:
                        text_conf = word_info[1]
                        if isinstance(text_conf, (tuple, list)) and len(text_conf) >= 1:
                            extracted_text_list.append(text_conf[0])
                            
        extracted_text = " ".join(extracted_text_list) if extracted_text_list else None
        logger.info(f"Optimized extraction for {doc_type} identifier ({best_class}): {extracted_text}")
        return extracted_text
        
    except Exception as e:
        logger.error(f"Error extracting linking identifier: {e}", exc_info=True)
    return None

def clean_id_number(val):
    """Cleans up the ID number/text for comparison by removing non-alphanumeric chars."""
    if not val or not isinstance(val, str):
        return ""
    # Filter only alphanumeric characters and uppercase
    return "".join(c.upper() for c in val if c.isalnum())

def validate_single_image(image_path, expected_type=None, expected_side=None, confidence_threshold=0.75):
    """
    Validates a single image against an expected document type and side.
    
    Returns:
        dict: The result containing is_valid, status, detected_type, detected_side, confidence, and message.
    """
    if not os.path.exists(image_path):
        return {
            "is_valid": False,
            "status": "error",
            "detected_type": None,
            "detected_side": None,
            "confidence": 0.0,
            "message": f"File does not exist: {image_path}"
        }
        
    try:
        classifier = load_yolo_model("Id_Classifier")
        image = cv2.imread(image_path)
        if image is None:
            return {
                "is_valid": False,
                "status": "error",
                "detected_type": None,
                "detected_side": None,
                "confidence": 0.0,
                "message": f"Failed to load image: {image_path}"
            }
            
        # Classify the ID type
        results = classifier(image)
        if not results or results[0].probs is None:
            return {
                "is_valid": False,
                "status": "unable_to_verify",
                "detected_type": None,
                "detected_side": None,
                "confidence": 0.0,
                "message": "Classification model returned no prediction results."
            }
            
        raw_class = results[0].names[results[0].probs.top1]
        confidence = results[0].probs.top1conf.item()
        
        logger.info(f"Classified {image_path} as '{raw_class}' with confidence {confidence:.2f}")
    except Exception as e:
        return {
            "is_valid": False,
            "status": "error",
            "detected_type": None,
            "detected_side": None,
            "confidence": 0.0,
            "message": f"Error running classifier: {str(e)}"
        }
        
    # 1. Check if classification confidence meets the threshold
    if confidence < confidence_threshold:
        return {
            "is_valid": False,
            "status": "unable_to_verify",
            "detected_type": None,
            "detected_side": None,
            "confidence": confidence,
            "message": f"Classification confidence too low ({confidence:.2f} < {confidence_threshold})"
        }
        
    # Map raw classification class to normalized type and side
    mapping = {
        "aadhar_front": ("Aadhaar", "front"),
        "aadhar_back": ("Aadhaar", "back"),
        "driving_license_front": ("Driving_License", "front"),
        "driving_license_back": ("Driving_License", "back"),
        "pan_card_front": ("Pan_Card", "front"),
        "passport": ("Passport", "front"),
        "voter_id": ("Voter_Id", None)  # Side resolved via custom model
    }
    
    if raw_class not in mapping:
        return {
            "is_valid": False,
            "status": "unable_to_verify",
            "detected_type": None,
            "detected_side": None,
            "confidence": confidence,
            "message": f"Unknown classification class: {raw_class}"
        }
        
    detected_type, detected_side = mapping[raw_class]
    
    # Resolve Voter ID side if generic voter_id class is returned
    if detected_type == "Voter_Id" and detected_side is None:
        detected_side = determine_voter_id_side(image_path)
        if detected_side is None:
            return {
                "is_valid": False,
                "status": "unable_to_verify",
                "detected_type": "Voter_Id",
                "detected_side": None,
                "confidence": confidence,
                "message": "Classified as Voter ID, but unable to determine front/back side confidently."
            }
            
    # Resolve/Correct Driving License side (classifier can confuse front/back)
    if detected_type == "Driving_License":
        resolved_side = determine_driving_license_side(image_path)
        if resolved_side:
            logger.info(f"Overriding DL side from classifier ({detected_side}) to resolved side ({resolved_side})")
            detected_side = resolved_side
            
    # Normalize expected type for comparison
    norm_expected_type = normalize_document_type(expected_type)
    
    # 2. Validate document type matches expected type
    if norm_expected_type and detected_type != norm_expected_type:
        return {
            "is_valid": False,
            "status": "mismatch",
            "detected_type": detected_type,
            "detected_side": detected_side,
            "confidence": confidence,
            "message": f"Document type mismatch: expected {norm_expected_type}, but detected {detected_type}"
        }
        
    # 3. Validate document side matches expected side
    if expected_side and detected_side != expected_side:
        return {
            "is_valid": False,
            "status": "mismatch",
            "detected_type": detected_type,
            "detected_side": detected_side,
            "confidence": confidence,
            "message": f"Document side mismatch: expected {expected_side}, but detected {detected_side}"
        }
        
    return {
        "is_valid": True,
        "status": "success",
        "detected_type": detected_type,
        "detected_side": detected_side,
        "confidence": confidence,
        "message": f"Verified successfully as {detected_type} {detected_side}"
    }

def validate_document(front_image_path=None, back_image_path=None, expected_type=None, confidence_threshold=0.75):
    """
    Validates uploaded document images against the expected document type and side.
    Supports single-side uploads (front-only, back-only) and dual-side uploads.
    
    For dual-side uploads:
    - Verifies that both sides belong to the same document type.
    - Runs detection and OCR to verify that both sides contain the same unique document ID.
    
    Args:
        front_image_path (str, optional): Path to the front image.
        back_image_path (str, optional): Path to the back image.
        expected_type (str): The document type claimed by the user.
        confidence_threshold (float): Minimum confidence threshold for verification.
        
    Returns:
        dict: Comprehensive validation report.
    """
    if not front_image_path and not back_image_path:
        return {
            "is_valid": False,
            "status": "error",
            "errors": ["At least one image path (front or back) must be provided."],
            "warnings": [],
            "details": {}
        }
        
    report = {
        "is_valid": True,
        "status": "success",
        "errors": [],
        "warnings": [],
        "details": {}
    }
    
    # --- 1. Single side validations ---
    if front_image_path:
        logger.info(f"Validating front image: {front_image_path}")
        front_res = validate_single_image(front_image_path, expected_type, "front", confidence_threshold)
        report["details"]["front"] = front_res
        if not front_res["is_valid"]:
            report["is_valid"] = False
            report["status"] = front_res["status"]
            report["errors"].append(f"Front side error: {front_res['message']}")
            
    if back_image_path:
        logger.info(f"Validating back image: {back_image_path}")
        back_res = validate_single_image(back_image_path, expected_type, "back", confidence_threshold)
        report["details"]["back"] = back_res
        if not back_res["is_valid"]:
            # If front was valid but back isn't, we override is_valid and status
            report["is_valid"] = False
            if report["status"] == "success" or report["status"] == "unable_to_verify":
                report["status"] = back_res["status"]
            report["errors"].append(f"Back side error: {back_res['message']}")
            
    # --- 2. Cross-side validations (if both front and back provided and valid) ---
    if front_image_path and back_image_path and report["is_valid"]:
        front_details = report["details"]["front"]
        back_details = report["details"]["back"]
        
        # Ensure they are of the same document type
        if front_details["detected_type"] != back_details["detected_type"]:
            report["is_valid"] = False
            report["status"] = "mismatch"
            report["errors"].append(
                f"Document type mismatch: Front is {front_details['detected_type']} but Back is {back_details['detected_type']}"
            )
        else:
            # Document types match. Now extract and check unique ID numbers
            doc_type = front_details["detected_type"]
            logger.info(f"Cross-linking front and back sides of {doc_type} using OCR...")
            
            front_id = extract_linking_identifier(front_image_path, doc_type)
            back_id = extract_linking_identifier(back_image_path, doc_type)
            
            clean_front = clean_id_number(front_id)
            clean_back = clean_id_number(back_id)
            
            report["details"]["cross_link"] = {
                "is_valid": True,
                "front_id_raw": front_id,
                "back_id_raw": back_id
            }
            
            if clean_front and clean_back:
                # Suffix-based matching rules:
                # For masked cards (e.g. front is "9281" last 4 digits, back is "343850059281")
                # Shorter must be a suffix of the longer one, and must be at least 4 digits.
                shorter = clean_front if len(clean_front) < len(clean_back) else clean_back
                longer = clean_back if len(clean_front) < len(clean_back) else clean_front
                
                if clean_front == clean_back:
                    logger.info("OCR unique identifiers matched exactly.")
                elif len(shorter) >= 4 and longer.endswith(shorter):
                    logger.info(f"OCR unique identifiers matched via suffix (masked ID). Suffix: {shorter}")
                else:
                    report["is_valid"] = False
                    report["status"] = "mismatch"
                    report["errors"].append(
                        f"Cross-link verification failed: Front identifier ({front_id}) does not match Back identifier ({back_id})."
                    )
                    report["details"]["cross_link"]["is_valid"] = False
            else:
                # We couldn't extract identifiers on both sides (normal if OCR fails on noisy/rotated images)
                # We do not fail the validation, but report a warning
                report["warnings"].append(
                    "Cross-link warning: Unique document identifier was not detected/readable on both sides. "
                    f"Front detected: '{front_id}', Back detected: '{back_id}'."
                )
                
    return report

if __name__ == "__main__":
    # Small test CLI for the validation module
    import argparse
    parser = argparse.ArgumentParser(description="Indian ID Validator - Validation Engine CLI")
    parser.add_argument("--front", help="Path to front image")
    parser.add_argument("--back", help="Path to back image")
    parser.add_argument("--expected-type", required=True, help="Expected document type (e.g. Aadhaar, Pan_Card, Voter_Id, Driving_License, Passport)")
    parser.add_argument("--threshold", type=float, default=0.75, help="Confidence threshold")
    
    args = parser.parse_args()
    
    result = validate_document(
        front_image_path=args.front,
        back_image_path=args.back,
        expected_type=args.expected_type,
        confidence_threshold=args.threshold
    )
    
    print("\n--- Validation Report ---")
    print(json.dumps(result, indent=4))
