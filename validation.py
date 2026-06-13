import os
import cv2
import json
import logging
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Import required components from the main pipeline
from inference import CONFIG, process_id, load_yolo_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import torch
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

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

# Replaced local load_yolo_model with import from inference.py for caching

def determine_voter_id_side(image_path):
    """
    Determines whether a Voter ID image is the front or back side.
    Since the Id_Classifier outputs a generic 'voter_id' class, this function
    uses the custom Voter_Id detection model to inspect detected features.
    """
    try:
        voter_model = load_yolo_model("Voter_Id")
        results = voter_model(image_path, device=DEVICE)
        
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

def determine_aadhaar_side(image_path):
    """
    Determines whether an Aadhaar image contains the front or back side by checking detected fields.
    For vertical e-Aadhaar letters (which contain both address and portrait), if front-specific
    fields (Name, DOB, Gender) are present, it resolves to "front".
    """
    try:
        aadhaar_model = load_yolo_model("Aadhaar")
        results = aadhaar_model(image_path, device=DEVICE)
        
        front_score = 0.0
        back_score = 0.0
        
        # Front indicators: Name, DOB, Gender
        # Back indicators: Address
        front_classes = {"Name", "DOB", "Gender"}
        back_classes = {"Address"}
        
        class_names = CONFIG["models"]["Aadhaar"]["classes"]
        
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
        
        logger.info(f"Aadhaar side determination scores -> Front: {front_score:.2f}, Back: {back_score:.2f}")
        
        # If front-specific fields are present, resolve to front (even if address is also present,
        # since a front side is contained in the image)
        if front_score > 0.5:
            return "front"
        elif back_score > 0.5:
            return "back"
    except Exception as e:
        logger.error(f"Error determining Aadhaar side: {e}")
    
    return None

def is_valid_aadhaar_number(text):
    if not text:
        return False
    clean = text.upper().replace(" ", "").replace("-", "")
    has_masking = any(c in clean for c in ["X", "*"])
    clean_no_mask = clean.replace("X", "").replace("*", "")
    
    if not clean_no_mask.isdigit():
        return False
        
    if has_masking:
        # Masked Aadhaar number: e.g. XXXX XXXX 9281, total length 8 to 12, non-masked is 4 digits
        return len(clean) >= 8 and len(clean) <= 12 and len(clean_no_mask) == 4
    else:
        # Full Aadhaar number: exactly 12 digits
        return len(clean) == 12

def determine_driving_license_side(image_path):
    """
    Determines whether a Driving License image is the front or back side.
    Uses a fast OCR keyword check (on a downscaled version of the image)
    which is extremely robust compared to bounding box class heuristics.
    """
    try:
        from inference import OCR
        import cv2
        
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        # Downscale to max width 512 for sub-200ms OCR execution
        h, w = img.shape[:2]
        if w > 512:
            scale = 512 / w
            img = cv2.resize(img, (512, int(h * scale)))
            
        padded = cv2.copyMakeBorder(img, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        ocr_res = OCR.ocr(padded)
        
        text_list = []
        if ocr_res:
            for item in ocr_res:
                if isinstance(item, dict):
                    text_list.extend(item.get("rec_texts", []))
                elif isinstance(item, list):
                    for word in item:
                        if isinstance(word, list) and len(word) >= 2:
                            text_list.append(word[1][0])
                            
        full_text = " ".join(text_list).lower()
        logger.info(f"DL side OCR text: {full_text}")
        
        back_keywords = ["address", "signature", "holder", "endorsement", "sign", "authority"]
        front_keywords = ["dob", "father", "blood", "licence", "license", "validity"]
        
        back_matches = sum(1 for kw in back_keywords if kw in full_text)
        front_matches = sum(1 for kw in front_keywords if kw in full_text)
        
        if "address" in full_text:
            return "back"
            
        if back_matches > front_matches:
            return "back"
        elif front_matches > back_matches:
            return "front"
            
    except Exception as e:
        logger.error(f"Error in OCR DL side resolution: {e}")
        
    # Fallback to YOLO model heuristic if OCR fails
    try:
        dl_model = load_yolo_model("Driving_License")
        results = dl_model(image_path, device=DEVICE)
        
        front_score = 0.0
        back_score = 0.0
        
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
                        
        if front_score > back_score and front_score > 0.3:
            return "front"
        elif back_score > front_score and back_score > 0.3:
            return "back"
    except Exception as e:
        logger.error(f"Error determining Driving License side: {e}")
    
    return None

def is_aadhaar_card(image):
    """
    Checks if the image contains structural features that uniquely identify an Aadhaar card.
    Useful for overriding classification errors where Aadhaar cards are misclassified as Voter ID.
    """
    try:
        model = load_yolo_model("Aadhaar")
        results = model(image, device=DEVICE)
        class_names = CONFIG["models"]["Aadhaar"]["classes"]
        for r in results:
            if not r.boxes:
                continue
            for box in r.boxes:
                cls_idx = int(box.cls[0].item())
                conf = box.conf[0].item()
                if cls_idx < len(class_names):
                    name = class_names[cls_idx]
                    # ONLY trust the Aadhaar number field, which is unique to Aadhaar layouts
                    if name == "Aadhaar" and conf >= 0.7:
                        # Verify the format of the detected Aadhaar number to prevent false positives on DL
                        h, w, _ = image.shape
                        x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                        x_min, y_min = max(0, x_min), max(0, y_min)
                        x_max, y_max = min(w, x_max), min(h, y_max)
                        region = image[y_min:y_max, x_min:x_max]
                        if region.size == 0:
                            continue
                            
                        padded = cv2.copyMakeBorder(region, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                        from inference import OCR
                        ocr_res = OCR.ocr(padded)
                        if ocr_res:
                            text_list = []
                            for item in ocr_res:
                                if isinstance(item, dict):
                                    text_list.extend(item.get("rec_texts", []))
                                elif isinstance(item, list):
                                    for word in item:
                                        if isinstance(word, list) and len(word) >= 2:
                                            text_list.append(word[1][0])
                            text = "".join(text_list).strip()
                            if is_valid_aadhaar_number(text):
                                logger.info(f"Confirmed as actual Aadhaar card (Aadhaar No format matches: '{text}')")
                                return True
    except Exception as e:
        logger.error(f"Error in is_aadhaar_card check: {e}")
    return False

def verify_identifier_presence(image, doc_type):
    """
    Verifies that the unique ID number field is present and contains readable text.
    Acts as a secure check to filter out random, out-of-domain, or blank images.
    """
    try:
        model = load_yolo_model(doc_type)
        results = model(image, device=DEVICE)
        class_names = CONFIG["models"][doc_type]["classes"]
        
        target_fields = {
            "Aadhaar": {"Aadhaar"},
            "Pan_Card": {"PAN"},
            "Driving_License": {"DL No"},
            "Voter_Id": {"Voter ID"},
            "Passport": {"Code", "MRZ1", "MRZ2"}
        }
        
        targets = target_fields.get(doc_type, set())
        
        for r in results:
            if not r.boxes:
                continue
            for box in r.boxes:
                cls_idx = int(box.cls[0].item())
                conf = box.conf[0].item()
                if cls_idx < len(class_names):
                    name = class_names[cls_idx]
                    if name in targets and conf >= 0.6:
                        h, w, _ = image.shape
                        x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                        x_min, y_min = max(0, x_min), max(0, y_min)
                        x_max, y_max = min(w, x_max), min(h, y_max)
                        region = image[y_min:y_max, x_min:x_max]
                        if region.size == 0:
                            continue
                            
                        padded = cv2.copyMakeBorder(region, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                        from inference import OCR
                        ocr_res = OCR.ocr(padded)
                        if ocr_res:
                            text_list = []
                            for item in ocr_res:
                                if isinstance(item, dict):
                                    text_list.extend(item.get("rec_texts", []))
                                elif isinstance(item, list):
                                    for word in item:
                                        if isinstance(word, list) and len(word) >= 2:
                                            text_list.append(word[1][0])
                            text = "".join(text_list).strip()
                            if len(text) >= 4:
                                logger.info(f"Verified identifier presence for {doc_type}: '{text}'")
                                return True
    except Exception as e:
        logger.error(f"Error verifying identifier presence: {e}")
    return False

def confirm_document_type(image, doc_type, threshold=0.5):
    """
    Validates if the image contains features of the expected document type.
    Serves as a fallback correction when the classifier misclassifies the document.
    """
    try:
        model = load_yolo_model(doc_type)
        results = model(image, device=DEVICE)
        
        class_names = CONFIG["models"][doc_type]["classes"]
        
        detections = []
        for result in results:
            if not result.boxes:
                continue
            for box in result.boxes:
                cls_idx = int(box.cls[0].item())
                conf = box.conf[0].item()
                if cls_idx < len(class_names):
                    detections.append((class_names[cls_idx], conf))
                    
        logger.info(f"confirm_document_type for '{doc_type}': detected fields {detections}")
        
        if doc_type == "Aadhaar":
            has_high_conf_key = any(name in ["Aadhaar", "Address"] and conf >= 0.7 for name, conf in detections)
            has_two_fields = len([conf for name, conf in detections if conf >= 0.5]) >= 2
            return has_high_conf_key or has_two_fields
            
        elif doc_type == "Pan_Card":
            has_key = any(name in ["PAN", "Pan Card"] and conf >= 0.6 for name, conf in detections)
            has_two = len([conf for name, conf in detections if conf >= 0.5]) >= 2
            return has_key or has_two
            
        elif doc_type == "Driving_License":
            has_key = any(name in ["DL No", "Vehicle Type"] and conf >= 0.6 for name, conf in detections)
            has_two = len([conf for name, conf in detections if conf >= 0.5]) >= 2
            return has_key or has_two
            
        elif doc_type == "Voter_Id":
            voter_keys = {"Voter ID", "Card Voter ID 1 Front", "Card Voter ID 1 Back", "Card Voter ID 2 Front", "Card Voter ID 2 Back"}
            has_key = any(name in voter_keys and conf >= 0.6 for name, conf in detections)
            has_two = len([conf for name, conf in detections if conf >= 0.5]) >= 2
            return has_key or has_two
            
        elif doc_type == "Passport":
            passport_keys = {"MRZ1", "MRZ2", "Code"}
            has_key = any(name in passport_keys and conf >= 0.6 for name, conf in detections)
            has_two = len([conf for name, conf in detections if conf >= 0.5]) >= 2
            return has_key or has_two
            
        return len([conf for name, conf in detections if conf >= threshold]) >= 1
    except Exception as e:
        logger.error(f"Error in confirm_document_type for {doc_type}: {e}")
        return False

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
        results = model(image_path, device=DEVICE)
        
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
        
        # Pad the crop (e.g. 15px white border) to aid OCR (10x faster than original giant black canvas)
        padded_img = cv2.copyMakeBorder(region_img, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        # 7. Run OCR
        ocr_result = OCR.ocr(padded_img)
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
        results = classifier(image, device=DEVICE)
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
    norm_expected_type = normalize_document_type(expected_type)
    
    # 1. First, check if it is actually an Aadhaar card (since classifier is prone to false positives on vertical Aadhaar)
    is_aadhaar = is_aadhaar_card(image)
    if is_aadhaar:
        if detected_type != "Aadhaar":
            logger.info(f"Overriding type from classifier ({detected_type}) to Aadhaar because Aadhaar structural fields were confidently detected.")
            detected_type = "Aadhaar"
            detected_side = None
            
    # 2. General type mismatch override fallback
    if norm_expected_type and detected_type != norm_expected_type:
        # If the document is verified as Aadhaar, but they expect something else, do NOT allow overriding to the expected type!
        if is_aadhaar:
            logger.info(f"Document confirmed as Aadhaar, but expected type is {norm_expected_type}. Disallowing override.")
        else:
            logger.info(f"Type mismatch detected (classifier: {detected_type}, expected: {norm_expected_type}). Running structure confirmation...")
            if confirm_document_type(image, norm_expected_type, threshold=0.5):
                logger.info(f"Overriding classified type from {detected_type} to expected type {norm_expected_type}")
                detected_type = norm_expected_type
                detected_side = None
                if detected_type in ["Pan_Card", "Passport"]:
                    detected_side = "front"
    
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
            
    # Resolve/Correct Aadhaar side (e.g. for vertical Aadhaar letters containing both sides)
    if detected_type == "Aadhaar":
        resolved_side = determine_aadhaar_side(image_path)
        if resolved_side:
            logger.info(f"Overriding Aadhaar side from classifier ({detected_side}) to resolved side ({resolved_side})")
            detected_side = resolved_side
            
    # Verify that the unique ID number field is present and readable on the document
    # For DL and Voter ID, we only require it on the front side.
    requires_id = True
    if detected_type in ["Driving_License", "Voter_Id"] and detected_side == "back":
        requires_id = False
        
    if requires_id:
        if not verify_identifier_presence(image, detected_type):
            logger.warning(f"Could not detect or read unique ID number on {image_path} for type {detected_type}.")
            return {
                "is_valid": False,
                "status": "unable_to_verify",
                "detected_type": detected_type,
                "detected_side": detected_side,
                "confidence": confidence,
                "message": f"Verification failed: Unable to detect or read the unique ID identifier (e.g. Aadhaar No, Voter EPIC, PAN, DL No) on the document."
            }

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
