import os
import cv2
import json
import logging
import re
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
    Returns None if the type does not represent a valid full document name.
    """
    if not expected_type or not isinstance(expected_type, str):
        return None
        
    # Convert to lowercase and strip all spaces, underscores, and hyphens
    clean = expected_type.lower().replace(" ", "").replace("_", "").replace("-", "")
    
    mapping = {
        "aadhaar": "Aadhaar",
        "aadhar": "Aadhaar",
        "pan": "Pan_Card",
        "pancard": "Pan_Card",
        "passport": "Passport",
        "voterid": "Voter_Id",
        "voter": "Voter_Id",
        "votercard": "Voter_Id",
        "drivinglicense": "Driving_License",
        "dl": "Driving_License"
    }
    return mapping.get(clean, None)

# Replaced local load_yolo_model with import from inference.py for caching

def determine_voter_id_side(image_path, cache=None):
    """
    Determines whether a Voter ID image is the front or back side.
    Since the Id_Classifier outputs a generic 'voter_id' class, this function
    uses the custom Voter_Id detection model to inspect detected features.
    """
    try:
        if cache is not None and (image_path, "voter_side") in cache["misc"]:
            return cache["misc"][(image_path, "voter_side")]
            
        results = None
        key_det = (image_path, "Voter_Id")
        if cache is not None:
            results = cache["detections"].get(key_det)
            
        if results is None:
            voter_model = load_yolo_model("Voter_Id")
            # If the image numpy array is already cached, run on it to avoid loading from disk again
            if cache is not None and image_path in cache["images"]:
                results = voter_model(cache["images"][image_path], device=DEVICE)
            else:
                results = voter_model(image_path, device=DEVICE)
            if cache is not None:
                cache["detections"][key_det] = results
        
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
        
        resolved_side = None
        if front_score > back_score and front_score > 0.3:
            resolved_side = "front"
        elif back_score > front_score and back_score > 0.3:
            resolved_side = "back"
            
        if cache is not None:
            cache["misc"][(image_path, "voter_side")] = resolved_side
        return resolved_side
    except Exception as e:
        logger.error(f"Error determining Voter ID side: {e}")
    
    return None

def determine_aadhaar_side(image_path, cache=None):
    """
    Determines whether an Aadhaar image contains the front or back side by checking detected fields.
    For vertical e-Aadhaar letters (which contain both address and portrait), if front-specific
    fields (Name, DOB, Gender) are present, it resolves to "front".
    """
    try:
        if cache is not None and (image_path, "aadhaar_side") in cache["misc"]:
            return cache["misc"][(image_path, "aadhaar_side")]
            
        results = None
        key_det = (image_path, "Aadhaar")
        if cache is not None:
            results = cache["detections"].get(key_det)
            
        if results is None:
            aadhaar_model = load_yolo_model("Aadhaar")
            if cache is not None and image_path in cache["images"]:
                results = aadhaar_model(cache["images"][image_path], device=DEVICE)
            else:
                results = aadhaar_model(image_path, device=DEVICE)
            if cache is not None:
                cache["detections"][key_det] = results
        
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
        
        resolved_side = None
        if front_score > 0.5:
            resolved_side = "front"
        elif back_score > 0.5:
            resolved_side = "back"
            
        if cache is not None:
            cache["misc"][(image_path, "aadhaar_side")] = resolved_side
        return resolved_side
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

def determine_driving_license_side(image_path, cache=None):
    """
    Determines whether a Driving License image is the front or back side.
    First checks YOLO detections for highly confident front-only features (DOB, Name).
    If ambiguous, falls back to a fast OCR keyword check on a downscaled version (320px).
    """
    try:
        if cache is not None and (image_path, "dl_side") in cache["misc"]:
            return cache["misc"][(image_path, "dl_side")]
    except Exception:
        pass

    resolved_side = None
    front_score = 0.0
    back_score = 0.0

    # 1. Run YOLO check first
    try:
        results = None
        key_det = (image_path, "Driving_License")
        if cache is not None:
            results = cache["detections"].get(key_det)
            
        if results is None:
            dl_model = load_yolo_model("Driving_License")
            img = None
            if cache is not None:
                img = cache["images"].get(image_path)
            if img is None:
                img = cv2.imread(image_path)
                if cache is not None and img is not None:
                    cache["images"][image_path] = img
            if img is not None:
                results = dl_model(img, device=DEVICE)
            else:
                results = dl_model(image_path, device=DEVICE)
            if cache is not None:
                cache["detections"][key_det] = results
                
        has_confident_front_field = False
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
                        if class_name in ["DOB", "Name"] and conf >= 0.5:
                            has_confident_front_field = True
                    elif class_name in back_classes:
                        back_score += conf

        # If we confidently detect Name/DOB and front score is dominant, shortcut to front
        if has_confident_front_field and front_score > back_score:
            resolved_side = "front"
            logger.info("DL side resolved to 'front' via YOLO Name/DOB shortcut.")
            
    except Exception as e:
        logger.error(f"Error in YOLO side check for Driving License: {e}")

    # 2. Fallback to fast OCR check if YOLO check is ambiguous/indicates back-side
    if resolved_side is None:
        # 2a. Try fast Tesseract OCR keyword check first
        try:
            import pytesseract
            import shutil
            import os
            
            tesseract_cmd = shutil.which("tesseract")
            if not tesseract_cmd:
                if os.path.exists("/opt/homebrew/bin/tesseract"):
                    tesseract_cmd = "/opt/homebrew/bin/tesseract"
                elif os.path.exists("/usr/local/bin/tesseract"):
                    tesseract_cmd = "/usr/local/bin/tesseract"
                    
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                
                img = None
                if cache is not None:
                    img = cache["images"].get(image_path)
                if img is None:
                    img = cv2.imread(image_path)
                    if cache is not None and img is not None:
                        cache["images"][image_path] = img
                        
                if img is not None:
                    h, w = img.shape[:2]
                    w_target = 512
                    if w > w_target:
                        scale = w_target / w
                        img_resized = cv2.resize(img, (w_target, int(h * scale)))
                    else:
                        img_resized = img
                        
                    # Preprocess for Tesseract: Convert to grayscale and apply adaptive thresholding
                    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                    
                    tess_text = pytesseract.image_to_string(adaptive).lower()
                    logger.info(f"DL side Tesseract text (512px): {tess_text.strip()[:100]}...")
                    
                    back_keywords = ["address", "signature", "holder", "endorsement", "sign", "authority", "form", "fom", "dto", "rto"]
                    front_keywords = ["dob", "father", "blood", "validity", "name", "husband", "s/d/w"]
                    
                    back_matches = sum(1 for kw in back_keywords if kw in tess_text)
                    front_matches = sum(1 for kw in front_keywords if kw in tess_text)
                    
                    if "address" in tess_text or back_matches > front_matches:
                        resolved_side = "back"
                    elif front_matches > back_matches:
                        resolved_side = "front"
                        
                    if resolved_side:
                        logger.info(f"DL side resolved to '{resolved_side}' via fast Tesseract OCR keyword check.")
        except Exception as e:
            logger.warning(f"Fast Tesseract DL side check skipped/failed: {e}")

        # 2b. Fallback to slow PaddleOCR check if Tesseract is not installed or failed to resolve
        if resolved_side is None:
            try:
                from inference import OCR
                
                img = None
                if cache is not None:
                    img = cache["images"].get(image_path)
                if img is None:
                    img = cv2.imread(image_path)
                    if cache is not None and img is not None:
                        cache["images"][image_path] = img
                        
                if img is not None:
                    # Downscale to max width 320 for sub-100ms OCR execution
                    h, w = img.shape[:2]
                    w_target = 320
                    if w > w_target:
                        scale = w_target / w
                        img_resized = cv2.resize(img, (w_target, int(h * scale)))
                    else:
                        img_resized = img
                        
                    padded = cv2.copyMakeBorder(img_resized, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                    
                    ocr_res = None
                    if cache is not None and (image_path, "full_ocr") in cache["misc"]:
                        ocr_res = cache["misc"][(image_path, "full_ocr")]
                        
                    if ocr_res is None:
                        ocr_res = OCR.ocr(padded)
                        if cache is not None:
                            cache["misc"][(image_path, "full_ocr")] = ocr_res
                    
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
                    logger.info(f"DL side OCR text (320px): {full_text}")
                    
                    back_keywords = ["address", "signature", "holder", "endorsement", "sign", "authority", "form", "fom", "dto", "rto"]
                    front_keywords = ["dob", "father", "blood", "validity", "name", "husband", "s/d/w"]
                    
                    back_matches = sum(1 for kw in back_keywords if kw in full_text)
                    front_matches = sum(1 for kw in front_keywords if kw in full_text)
                    
                    if "address" in full_text:
                        resolved_side = "back"
                    elif back_matches > front_matches:
                        resolved_side = "back"
                    elif front_matches > back_matches:
                        resolved_side = "front"
            except Exception as e:
                logger.error(f"Error in OCR DL side resolution: {e}")

    # 3. Final fallback to YOLO relative scores
    if resolved_side is None:
        try:
            if front_score > back_score and front_score > 0.3:
                resolved_side = "front"
            elif back_score > front_score and back_score > 0.3:
                resolved_side = "back"
        except Exception:
            pass

    if resolved_side is not None and cache is not None:
        cache["misc"][(image_path, "dl_side")] = resolved_side
    return resolved_side

def is_aadhaar_card(image, image_path=None, cache=None, threshold=0.7):
    """
    Checks if the image contains structural features that uniquely identify an Aadhaar card.
    Useful for overriding classification errors where Aadhaar cards are misclassified as Voter ID.
    """
    try:
        cache_key = (image_path if image_path else id(image), "is_aadhaar")
        if cache is not None and cache_key in cache["misc"]:
            return cache["misc"][cache_key]
            
        results = None
        key_det = (image_path if image_path else id(image), "Aadhaar")
        if cache is not None:
            results = cache["detections"].get(key_det)
            
        if results is None:
            model = load_yolo_model("Aadhaar")
            results = model(image, device=DEVICE)
            if cache is not None:
                cache["detections"][key_det] = results
                
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
                    if name == "Aadhaar" and conf >= threshold:
                        # Verify the format of the detected Aadhaar number to prevent false positives on DL
                        h, w, _ = image.shape
                        box_coords = tuple(map(int, box.xyxy[0].tolist()))
                        ocr_key = (image_path if image_path else id(image), box_coords)
                        
                        text = None
                        if cache is not None:
                            text = cache["ocr_crops"].get(ocr_key)
                            
                        if text is None:
                            x_min, y_min, x_max, y_max = box_coords
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
                            else:
                                text = ""
                            if cache is not None:
                                cache["ocr_crops"][ocr_key] = text
                                
                        if is_valid_aadhaar_number(text):
                            logger.info(f"Confirmed as actual Aadhaar card (Aadhaar No format matches: '{text}')")
                            if cache is not None:
                                cache["misc"][cache_key] = True
                            return True
    except Exception as e:
        logger.error(f"Error in is_aadhaar_card check: {e}")
        
    if cache is not None:
        cache["misc"][cache_key] = False
    return False

def search_identifier_in_text(text, doc_type):
    """
    Searches for a valid document identifier pattern in the raw OCR text.
    Acts as a secure check when YOLO detection fails to locate the bounding box.
    """
    if not text:
        return False
    text = text.upper()
    
    if doc_type == "Aadhaar":
        # Match standard 12-digit number (with optional spaces/hyphens) or masked version
        patterns = [
            r'\b\d{4}\s*\d{4}\s*\d{4}\b',
            r'\b[X\*]{4}\s*[X\*]{4}\s*\d{4}\b',
            r'\b\d{12}\b'
        ]
        return any(re.search(pat, text) for pat in patterns)
        
    elif doc_type == "Pan_Card":
        # Match 5 letters + 4 digits + 1 letter
        pattern = r'\b[A-Z]{5}\d{4}[A-Z]\b'
        return bool(re.search(pattern, text))
        
    elif doc_type == "Driving_License":
        # Match standard DL: e.g. DL14 20110012345, DL-1420110012345, JH09 20210020533, RJ14C20220018637
        patterns = [
            r'[A-Z]{2}\d{2}[A-Z]?\s*\d{4}\s*\d{7}',
            r'[A-Z]{2}-\d{2}-\d{11}',
            r'[A-Z]{2}\d{13}'
        ]
        return any(re.search(pat, text) for pat in patterns)
        
    elif doc_type == "Voter_Id":
        # Match EPIC format: 3 letters + 7 digits, or slash-separated state formats e.g. DL/01/001/012345
        patterns = [
            r'\b[A-Z]{3}\d{7}\b',
            r'[A-Z]{2,3}/\d+(?:/\d+)+'
        ]
        return any(re.search(pat, text) for pat in patterns)
        
    elif doc_type == "Passport":
        # Match 1 letter + 7 digits, or MRZ lines
        patterns = [
            r'\b[A-Z]\d{7}\b',
            r'P<[A-Z]{3}[A-Z_]+<<'
        ]
        return any(re.search(pat, text) for pat in patterns)
        
    return False

def extract_identifier_from_text(text, doc_type):
    """
    Extracts the unique ID number from full-image OCR text using pattern matching.
    """
    if not text:
        return None
    text = text.upper()
    
    if doc_type == "Aadhaar":
        # Find 12 digits or masked pattern
        patterns = [
            r'\b(\d{4}\s\d{4}\s\d{4})\b',
            r'\b([X\*]{4}\s[X\*]{4}\s\d{4})\b',
            r'\b(\d{12})\b'
        ]
        for pat in patterns:
            match = re.search(pat, text)
            if match:
                return match.group(1)
                
    elif doc_type == "Pan_Card":
        # Find 5 letters + 4 digits + 1 letter
        pattern = r'\b([A-Z]{5}\d{4}[A-Z])\b'
        match = re.search(pattern, text)
        if match:
            return match.group(1)
            
    elif doc_type == "Driving_License":
        # Find standard DL: e.g. DL14 20110012345, DL-1420110012345, JH09 20210020533, RJ14C20220018637
        patterns = [
            r'([A-Z]{2}\d{2}[A-Z]?\s*\d{4}\s*\d{7})',
            r'([A-Z]{2}-\d{2}-\d{11})',
            r'([A-Z]{2}\d{13})'
        ]
        for pat in patterns:
            match = re.search(pat, text)
            if match:
                return match.group(1)
                
    elif doc_type == "Voter_Id":
        # Find EPIC format
        patterns = [
            r'\b([A-Z]{3}\d{7})\b',
            r'([A-Z]{2,3}/\d+(?:/\d+)+)'
        ]
        for pat in patterns:
            match = re.search(pat, text)
            if match:
                return match.group(1)
                
    elif doc_type == "Passport":
        # Find 1 letter + 7 digits
        pattern = r'\b([A-Z]\d{7})\b'
        match = re.search(pattern, text)
        if match:
            return match.group(1)
            
    return None

def verify_identifier_presence(image, doc_type, image_path=None, cache=None):
    """
    Verifies that the unique ID number field is present and contains readable text.
    Acts as a secure check to filter out random, out-of-domain, or blank images.
    """
    try:
        cache_key = (image_path if image_path else id(image), f"verify_id_{doc_type}")
        if cache is not None and cache_key in cache["misc"]:
            return cache["misc"][cache_key]
            
        results = None
        key_det = (image_path if image_path else id(image), doc_type)
        if cache is not None:
            results = cache["detections"].get(key_det)
            
        if results is None:
            model = load_yolo_model(doc_type)
            results = model(image, device=DEVICE)
            if cache is not None:
                cache["detections"][key_det] = results
                
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
                        box_coords = tuple(map(int, box.xyxy[0].tolist()))
                        ocr_key = (image_path if image_path else id(image), box_coords)
                        
                        text = None
                        if cache is not None:
                            text = cache["ocr_crops"].get(ocr_key)
                            
                        if text is None:
                            x_min, y_min, x_max, y_max = box_coords
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
                            else:
                                text = ""
                            if cache is not None:
                                cache["ocr_crops"][ocr_key] = text
                                
                        if len(text) >= 4:
                            logger.info(f"Verified identifier presence for {doc_type}: '{text}'")
                            if cache is not None:
                                cache["misc"][cache_key] = True
                            return True
        # Fallback: If YOLO model fails to detect the identifier box, check the full image OCR text for expected patterns.
        # This handles layout variations across different states.
        logger.info(f"YOLO detection did not find confident identifier class for {doc_type}. Running full-image OCR search fallback...")
        
        # Fetch full image OCR results (check cache first)
        ocr_res = None
        if cache is not None and (image_path if image_path else id(image), "full_ocr") in cache["misc"]:
            ocr_res = cache["misc"][(image_path if image_path else id(image), "full_ocr")]
            
        if ocr_res is None:
            # Downscale image to max width 320 for sub-100ms OCR execution
            h, w = image.shape[:2]
            w_target = 320
            if w > w_target:
                scale = w_target / w
                img_resized = cv2.resize(image, (w_target, int(h * scale)))
            else:
                img_resized = image
            padded = cv2.copyMakeBorder(img_resized, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            from inference import OCR
            ocr_res = OCR.ocr(padded)
            if cache is not None:
                cache["misc"][(image_path if image_path else id(image), "full_ocr")] = ocr_res
                
        text_list = []
        if ocr_res:
            for item in ocr_res:
                if isinstance(item, dict):
                    text_list.extend(item.get("rec_texts", []))
                elif isinstance(item, list):
                    for word in item:
                        if isinstance(word, list) and len(word) >= 2:
                            text_list.append(word[1][0])
                            
        full_text = " ".join(text_list)
        
        if search_identifier_in_text(full_text, doc_type):
            logger.info(f"Verified identifier presence for {doc_type} via full-image OCR search fallback. Text: '{full_text}'")
            if cache is not None:
                cache["misc"][cache_key] = True
            return True
            
    except Exception as e:
        logger.error(f"Error verifying identifier presence: {e}")
        
    if cache is not None:
        cache["misc"][cache_key] = False
    return False

def confirm_document_type(image, doc_type, threshold=0.5, image_path=None, cache=None):
    """
    Validates if the image contains features of the expected document type.
    Serves as a fallback correction when the classifier misclassifies the document.
    """
    try:
        cache_key = (image_path if image_path else id(image), f"confirm_{doc_type}")
        if cache is not None and cache_key in cache["misc"]:
            return cache["misc"][cache_key]
            
        results = None
        key_det = (image_path if image_path else id(image), doc_type)
        if cache is not None:
            results = cache["detections"].get(key_det)
            
        if results is None:
            model = load_yolo_model(doc_type)
            results = model(image, device=DEVICE)
            if cache is not None:
                cache["detections"][key_det] = results
        
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
        
        result_bool = False
        if doc_type == "Aadhaar":
            has_high_conf_key = any(name in ["Aadhaar"] and conf >= 0.7 for name, conf in detections)
            has_two_fields = len([conf for name, conf in detections if conf >= 0.5]) >= 2
            result_bool = has_high_conf_key or has_two_fields
            
        elif doc_type == "Pan_Card":
            has_key = any(name in ["PAN", "Pan Card"] and conf >= 0.6 for name, conf in detections)
            has_two = len([conf for name, conf in detections if conf >= 0.5]) >= 2
            result_bool = has_key or has_two
            
        elif doc_type == "Driving_License":
            has_key = any(name in ["DL No", "Vehicle Type"] and conf >= 0.6 for name, conf in detections)
            has_two = len([conf for name, conf in detections if conf >= 0.5]) >= 2
            result_bool = has_key or has_two
            
        elif doc_type == "Voter_Id":
            voter_keys = {"Voter ID", "Card Voter ID 1 Front", "Card Voter ID 1 Back", "Card Voter ID 2 Front", "Card Voter ID 2 Back", "Symbol", "Election", "Date of Issue"}
            has_key = any(name in voter_keys and conf >= 0.6 for name, conf in detections)
            has_two = False
            valid_detections = [name for name, conf in detections if conf >= 0.5]
            if len(valid_detections) >= 2:
                has_two = any(name in voter_keys for name in valid_detections)
            result_bool = has_key or has_two
            
        elif doc_type == "Passport":
            passport_keys = {"MRZ1", "MRZ2", "Code"}
            has_key = any(name in passport_keys and conf >= 0.6 for name, conf in detections)
            has_two = len([conf for name, conf in detections if conf >= 0.5]) >= 2
            result_bool = has_key or has_two
            
        else:
            result_bool = len([conf for name, conf in detections if conf >= threshold]) >= 1
            
        if not result_bool:
            # Fallback: If YOLO model fails to confirm structurally, check full-image OCR text for high-confidence keywords.
            logger.info(f"YOLO structural confirmation failed for {doc_type}. Running full-image OCR keyword fallback...")
            
            # Fetch full image OCR results
            ocr_res = None
            if cache is not None and (image_path if image_path else id(image), "full_ocr") in cache["misc"]:
                ocr_res = cache["misc"][(image_path if image_path else id(image), "full_ocr")]
                
            if ocr_res is None:
                h, w = image.shape[:2]
                w_target = 320
                if w > w_target:
                    scale = w_target / w
                    img_resized = cv2.resize(image, (w_target, int(h * scale)))
                else:
                    img_resized = image
                padded = cv2.copyMakeBorder(img_resized, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                from inference import OCR
                ocr_res = OCR.ocr(padded)
                if cache is not None:
                    cache["misc"][(image_path if image_path else id(image), "full_ocr")] = ocr_res
                    
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
            
            if doc_type == "Aadhaar":
                keywords = ["aadhaar", "uidai", "government of india", "enrollment", "unique identification", "आधार", "भारत सरकार"]
                result_bool = any(kw in full_text for kw in keywords)
                
            elif doc_type == "Pan_Card":
                keywords = ["pan", "permanent account", "income tax", "department", "govt of india", "incometaxindia", "आयकर"]
                result_bool = any(kw in full_text for kw in keywords)
                
            elif doc_type == "Driving_License":
                keywords = ["driving", "licence", "license", "form 7", "fom 7", "form-7", "endorsement", "transport", "rto", "dto", "motor", "vehicle", "cov"]
                result_bool = any(kw in full_text for kw in keywords)
                
            elif doc_type == "Voter_Id":
                keywords = ["election commission", "electoral", "voter", "epic", "identity card", "भारत निर्वाचन आयोग", "पहचान पत्र"]
                result_bool = any(kw in full_text for kw in keywords)
                
            elif doc_type == "Passport":
                keywords = ["passport", "republic of india", "भारत गणराज्य", "mrz"]
                result_bool = any(kw in full_text for kw in keywords)
                
        if cache is not None:
            cache["misc"][cache_key] = result_bool
        return result_bool
    except Exception as e:
        logger.error(f"Error in confirm_document_type for {doc_type}: {e}")
        return False

def extract_linking_identifier(image_path, doc_type, cache=None):
    """
    Optimized function to extract ONLY the unique ID number field,
    avoiding OCR on other fields (Name, Address, DOB, etc.) to save time.
    """
    try:
        import numpy as np
        from inference import preprocess_image, OCR
        
        # Load image (check cache first)
        image = None
        if cache is not None:
            image = cache["images"].get(image_path)
        if image is None:
            image = cv2.imread(image_path)
            if cache is not None and image is not None:
                cache["images"][image_path] = image
                
        if image is None:
            return None
            
        h, w, _ = image.shape
        
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
        
        # 2. Load the detection model (check cache first)
        results = None
        key_det = (image_path, doc_type)
        if cache is not None:
            results = cache["detections"].get(key_det)
            
        if results is None:
            model = load_yolo_model(doc_type)
            results = model(image, device=DEVICE)
            if cache is not None:
                cache["detections"][key_det] = results
                
        class_names = CONFIG["models"][doc_type]["classes"]
        
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
            logger.info(f"Target identifier class {target_classes} not detected in {image_path}. Running full-image OCR extraction fallback...")
            
            # Fetch full image OCR results
            ocr_res = None
            if cache is not None and (image_path, "full_ocr") in cache["misc"]:
                ocr_res = cache["misc"][(image_path, "full_ocr")]
                
            if ocr_res is None:
                h, w = image.shape[:2]
                w_target = 320
                if w > w_target:
                    scale = w_target / w
                    img_resized = cv2.resize(image, (w_target, int(h * scale)))
                else:
                    img_resized = image
                padded = cv2.copyMakeBorder(img_resized, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[255, 255, 255])
                ocr_res = OCR.ocr(padded)
                if cache is not None:
                    cache["misc"][(image_path, "full_ocr")] = ocr_res
                    
            text_list = []
            if ocr_res:
                for item in ocr_res:
                    if isinstance(item, dict):
                        text_list.extend(item.get("rec_texts", []))
                    elif isinstance(item, list):
                        for word in item:
                            if isinstance(word, list) and len(word) >= 2:
                                text_list.append(word[1][0])
                                
            full_text = " ".join(text_list)
            extracted_id = extract_identifier_from_text(full_text, doc_type)
            if extracted_id:
                logger.info(f"Extracted {doc_type} identifier via full-image OCR fallback: '{extracted_id}'")
                return extracted_id
                
            return None
            
        # Check cache for OCR crop result
        best_box_tuple = tuple(map(int, best_box))
        ocr_key = (image_path, best_box_tuple)
        if cache is not None:
            cached_text = cache["ocr_crops"].get(ocr_key)
            if cached_text is not None:
                logger.info(f"Optimized extraction (Cached) for {doc_type} identifier ({best_class}): {cached_text}")
                return cached_text
                
        # 3. Crop the single best bounding box
        x_min, y_min, x_max, y_max = best_box_tuple
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(w, x_max), min(h, y_max)
        
        region_img = image[y_min:y_max, x_min:x_max]
        if region_img.size == 0:
            return None
            
        # 4. Preprocess ONLY this crop
        region_img = preprocess_image(region_img)
        
        # Pad the crop (e.g. 15px white border) to aid OCR (10x faster than original giant black canvas)
        padded_img = cv2.copyMakeBorder(region_img, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        
        # 5. Run OCR
        ocr_result = OCR.ocr(padded_img)
        if not ocr_result:
            if cache is not None:
                cache["ocr_crops"][ocr_key] = ""
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
                            
        extracted_text = " ".join(extracted_text_list) if extracted_text_list else ""
        logger.info(f"Optimized extraction for {doc_type} identifier ({best_class}): {extracted_text}")
        
        if cache is not None:
            cache["ocr_crops"][ocr_key] = extracted_text
            
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

def validate_single_image(image_path, expected_type=None, expected_side=None, confidence_threshold=0.75, cache=None):
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
        
        # Load image (check cache first)
        image = None
        if cache is not None:
            image = cache["images"].get(image_path)
        if image is None:
            image = cv2.imread(image_path)
            if cache is not None and image is not None:
                cache["images"][image_path] = image
                
        if image is None:
            return {
                "is_valid": False,
                "status": "error",
                "detected_type": None,
                "detected_side": None,
                "confidence": 0.0,
                "message": f"Failed to load image: {image_path}"
            }
            
        # Automatic rotation detection and correction (Layer 1: Tesseract OSD)
        corrected_rotation = None
        try:
            import pytesseract
            import shutil
            
            tesseract_cmd = shutil.which("tesseract")
            if not tesseract_cmd:
                if os.path.exists("/opt/homebrew/bin/tesseract"):
                    tesseract_cmd = "/opt/homebrew/bin/tesseract"
                elif os.path.exists("/usr/local/bin/tesseract"):
                    tesseract_cmd = "/usr/local/bin/tesseract"
                    
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                osd = pytesseract.image_to_osd(image)
                rotate_angle = 0
                for line in osd.split("\n"):
                    if "Rotate:" in line:
                        rotate_angle = int(line.split(":")[1].strip())
                        break
                        
                if rotate_angle in [90, 180, 270]:
                    logger.info(f"Tesseract OSD detected document rotation. Need to rotate by {rotate_angle} degrees clockwise to align upright.")
                    if rotate_angle == 90:
                        corrected_rotation = cv2.ROTATE_90_CLOCKWISE
                    elif rotate_angle == 180:
                        corrected_rotation = cv2.ROTATE_180
                    elif rotate_angle == 270:
                        corrected_rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        except Exception as e:
            logger.info(f"Tesseract OSD orientation detection check skipped/failed: {e}")
            
        if corrected_rotation is not None:
            logger.info(f"Rotating image to upright orientation using OSD result (flag {corrected_rotation})")
            image = cv2.rotate(image, corrected_rotation)
            if cache is not None:
                cache["images"][image_path] = image
            
        # Classify the ID type
        results = classifier(image, device=DEVICE)
        
        raw_class = None
        confidence = 0.0
        if results and results[0].probs is not None:
            raw_class = results[0].names[results[0].probs.top1]
            confidence = results[0].probs.top1conf.item()
            
        logger.info(f"Initial classification for {image_path} (0 deg): '{raw_class}' with confidence {confidence:.2f}")
        
        # Automatic rotation detection and correction
        if confidence < confidence_threshold:
            logger.info(f"Confidence {confidence:.2f} is below threshold {confidence_threshold}. Running rotation correction check...")
            rotations = [
                (cv2.ROTATE_90_CLOCKWISE, 90),
                (cv2.ROTATE_180, 180),
                (cv2.ROTATE_90_COUNTERCLOCKWISE, 270)
            ]
            best_rotation = None
            best_raw_class = raw_class
            best_confidence = confidence
            best_results = results
            
            for rot_flag, angle in rotations:
                rotated_img = cv2.rotate(image, rot_flag)
                rot_results = classifier(rotated_img, device=DEVICE)
                if rot_results and rot_results[0].probs is not None:
                    rot_class = rot_results[0].names[rot_results[0].probs.top1]
                    rot_conf = rot_results[0].probs.top1conf.item()
                    logger.info(f"Rotation check ({angle} deg): '{rot_class}' with confidence {rot_conf:.2f}")
                    if rot_conf > best_confidence:
                        best_confidence = rot_conf
                        best_raw_class = rot_class
                        best_rotation = rot_flag
                        best_results = rot_results
            
            if best_rotation is not None and best_confidence >= confidence_threshold:
                logger.info(f"Correcting image rotation. Best orientation found at rotation flag {best_rotation} with confidence {best_confidence:.2f}")
                image = cv2.rotate(image, best_rotation)
                raw_class = best_raw_class
                confidence = best_confidence
                results = best_results
                if cache is not None:
                    cache["images"][image_path] = image
                    # Clear geometry-dependent cache keys
                    for k in [(image_path, "Driving_License"), (image_path, "Aadhaar"), (image_path, "Pan_Card"), (image_path, "Voter_Id"), (image_path, "Passport")]:
                        if k in cache["detections"]:
                            del cache["detections"][k]
                    if (image_path, "full_ocr") in cache["misc"]:
                        del cache["misc"][(image_path, "full_ocr")]
                    if (image_path, "dl_side") in cache["misc"]:
                        del cache["misc"][(image_path, "dl_side")]
                        
        if not raw_class:
            return {
                "is_valid": False,
                "status": "unable_to_verify",
                "detected_type": None,
                "detected_side": None,
                "confidence": 0.0,
                "message": "Classification model returned no prediction results."
            }
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
            "message": "We could not confidently identify this image as a valid ID card. Please ensure the image is clear, well-lit, uncropped, and not a random object or scenery."
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
    
    # Automatic rotation detection and correction (Layer 3: YOLO layout confirmation)
    # If the classifier confidently detected a document type, double-check its layout orientation.
    # Note: Passports don't have separate front/back YOLO layout models (we use Passport model for both).
    # If confirm_document_type fails in current orientation, check other rotations.
    is_upright = confirm_document_type(image, detected_type, threshold=0.4, image_path=image_path, cache=cache)
    if not is_upright:
        logger.info(f"YOLO layout confirmation failed for {detected_type} in current orientation on {image_path}. Running rotation check...")
        rotations = [
            (cv2.ROTATE_90_CLOCKWISE, 90),
            (cv2.ROTATE_180, 180),
            (cv2.ROTATE_90_COUNTERCLOCKWISE, 270)
        ]
        for rot_flag, angle in rotations:
            rotated_img = cv2.rotate(image, rot_flag)
            # Clear cache entries temporarily for this specific check to avoid returning 0-deg cached predictions
            if cache is not None:
                key_det = (image_path, detected_type)
                if key_det in cache["detections"]:
                    del cache["detections"][key_det]
                confirm_key = (image_path, f"confirm_{detected_type}")
                if confirm_key in cache["misc"]:
                    del cache["misc"][confirm_key]
                    
            if confirm_document_type(rotated_img, detected_type, threshold=0.4, image_path=image_path, cache=cache):
                logger.info(f"Correcting image rotation based on YOLO layout confirmation. Best orientation found at {angle} degrees clockwise.")
                image = rotated_img
                if cache is not None:
                    cache["images"][image_path] = image
                    # Clear all other geometry-dependent cache keys
                    for k in list(cache["detections"].keys()):
                        if k[0] == image_path:
                            del cache["detections"][k]
                    for k in list(cache["misc"].keys()):
                        if k[0] == image_path:
                            del cache["misc"][k]
                    for k in list(cache["ocr_crops"].keys()):
                        if k[0] == image_path:
                            del cache["ocr_crops"][k]
                break
    
    # 1. First, check if it is actually an Aadhaar card (since classifier is prone to false positives on vertical Aadhaar)
    is_aadhaar = False
    # Only run is_aadhaar check if detected type is Aadhaar, Voter_Id, or classification confidence is low
    if detected_type in ["Aadhaar", "Voter_Id"] or confidence < 0.80:
        is_aadhaar = is_aadhaar_card(image, image_path=image_path, cache=cache, threshold=0.7)
        if is_aadhaar:
            if detected_type != "Aadhaar":
                logger.info(f"Overriding type from classifier ({detected_type}) to Aadhaar because Aadhaar structural fields were confidently detected.")
                detected_type = "Aadhaar"
                detected_side = None
            
    # 2. General type mismatch override fallback
    if norm_expected_type and detected_type != norm_expected_type:
        # If the document is verified as Aadhaar, but they expect something else, do NOT allow overriding to the expected type!
        # Also, do not allow overriding highly confident classifier predictions to DL, PAN, or Passport to prevent false positive overrides.
        can_override = True
        if is_aadhaar:
            can_override = False
            logger.info(f"Document confirmed as Aadhaar, but expected type is {norm_expected_type}. Disallowing override.")
        elif confidence >= 0.90:
            # Allow overrides between Aadhaar and Voter_Id due to classifier confusion on vertical/card layouts
            confusable = (detected_type in ["Aadhaar", "Voter_Id"] and norm_expected_type in ["Aadhaar", "Voter_Id"])
            if not confusable:
                can_override = False
                logger.info(f"Trusting highly confident classifier prediction '{detected_type}' ({confidence:.2f}). Disallowing override to '{norm_expected_type}'.")
            
        if can_override:
            logger.info(f"Type mismatch detected (classifier: {detected_type}, expected: {norm_expected_type}). Running structure confirmation...")
            if confirm_document_type(image, norm_expected_type, threshold=0.5, image_path=image_path, cache=cache):
                logger.info(f"Overriding classified type from {detected_type} to expected type {norm_expected_type}")
                detected_type = norm_expected_type
                detected_side = None
                if detected_type in ["Pan_Card", "Passport"]:
                    detected_side = "front"
    
    # Resolve Voter ID side if generic voter_id class is returned
    if detected_type == "Voter_Id" and detected_side is None:
        detected_side = determine_voter_id_side(image_path, cache=cache)
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
        resolved_side = determine_driving_license_side(image_path, cache=cache)
        if resolved_side:
            logger.info(f"Overriding DL side from classifier ({detected_side}) to resolved side ({resolved_side})")
            detected_side = resolved_side
            
    # Resolve/Correct Aadhaar side (e.g. for vertical Aadhaar letters containing both sides)
    if detected_type == "Aadhaar":
        resolved_side = determine_aadhaar_side(image_path, cache=cache)
        if resolved_side:
            logger.info(f"Overriding Aadhaar side from classifier ({detected_side}) to resolved side ({resolved_side})")
            detected_side = resolved_side
            
    # Normalize expected type for comparison
    # (Since norm_expected_type was defined earlier, we don't redefine it)
    
    # 2. Validate document type matches expected type
    if norm_expected_type and detected_type != norm_expected_type:
        # If the detected type does not match the expected type, verify that the mismatched
        # document is actually a valid card of that detected type (for front sides)
        # to avoid falsely claiming we detected a Voter ID/DL/PAN on a random meme or scenery image.
        if detected_type is not None:
            side_to_check = detected_side
            if detected_type == "Voter_Id" and side_to_check is None:
                side_to_check = determine_voter_id_side(image_path, cache=cache)
                
            req_id_mismatch = True
            if detected_type in ["Driving_License", "Voter_Id", "Aadhaar"] and side_to_check == "back":
                req_id_mismatch = False
                
            mismatch_verified = False
            # Check initial orientation
            if req_id_mismatch:
                if verify_identifier_presence(image, detected_type, image_path=image_path, cache=cache):
                    mismatch_verified = True
            else:
                if confirm_document_type(image, detected_type, threshold=0.5, image_path=image_path, cache=cache):
                    mismatch_verified = True
                    
            if not mismatch_verified:
                logger.info(f"Mismatched detected type verification failed in current orientation on {image_path}. Running rotation fallback check...")
                rotations = [
                    (cv2.ROTATE_90_CLOCKWISE, 90),
                    (cv2.ROTATE_180, 180),
                    (cv2.ROTATE_90_COUNTERCLOCKWISE, 270)
                ]
                for rot_flag, angle in rotations:
                    rotated_img = cv2.rotate(image, rot_flag)
                    # Update cache temporarily
                    if cache is not None:
                        cache["images"][image_path] = rotated_img
                        for k in list(cache["detections"].keys()):
                            if k[0] == image_path:
                                del cache["detections"][k]
                        for k in list(cache["misc"].keys()):
                            if k[0] == image_path:
                                del cache["misc"][k]
                        for k in list(cache["ocr_crops"].keys()):
                            if k[0] == image_path:
                                del cache["ocr_crops"][k]
                                
                    if req_id_mismatch:
                        if verify_identifier_presence(rotated_img, detected_type, image_path=image_path, cache=cache):
                            logger.info(f"Correcting image rotation based on mismatched ID verification. Best orientation found at {angle} degrees clockwise.")
                            image = rotated_img
                            mismatch_verified = True
                            break
                    else:
                        if confirm_document_type(rotated_img, detected_type, threshold=0.5, image_path=image_path, cache=cache):
                            logger.info(f"Correcting image rotation based on mismatched back confirmation. Best orientation found at {angle} degrees clockwise.")
                            image = rotated_img
                            mismatch_verified = True
                            break
                            
            if not mismatch_verified:
                # Restore original image
                if cache is not None:
                    original_img = cv2.imread(image_path)
                    if original_img is not None:
                        cache["images"][image_path] = original_img
                    for k in list(cache["detections"].keys()):
                        if k[0] == image_path:
                            del cache["detections"][k]
                    for k in list(cache["misc"].keys()):
                        if k[0] == image_path:
                            del cache["misc"][k]
                    for k in list(cache["ocr_crops"].keys()):
                        if k[0] == image_path:
                            del cache["ocr_crops"][k]
                            
                logger.info(f"Mismatched detected type '{detected_type}' failed unique identifier check/back side confirmation under all rotations. Reclassifying as None to prevent false positive mismatch reports.")
                detected_type = None
                detected_side = None

        if detected_type is None:
            return {
                "is_valid": False,
                "status": "unable_to_verify",
                "detected_type": None,
                "detected_side": None,
                "confidence": confidence,
                "message": "We could not confidently identify this image as a valid ID card. Please ensure the image is clear, well-lit, uncropped, and not a random object or scenery."
            }

        return {
            "is_valid": False,
            "status": "mismatch",
            "detected_type": detected_type,
            "detected_side": detected_side,
            "confidence": confidence,
            "message": f"Document type mismatch: expected {norm_expected_type}, but detected {detected_type}"
        }
        
    # Verify that the unique ID number field is present and readable on the document
    # For DL, Voter ID, and Aadhaar, we only require it on the front side.
    requires_id = True
    if detected_type in ["Driving_License", "Voter_Id", "Aadhaar"] and detected_side == "back":
        requires_id = False
        
    verification_success = False
    if requires_id:
        if verify_identifier_presence(image, detected_type, image_path=image_path, cache=cache):
            verification_success = True
        else:
            logger.info(f"Unique ID presence verification failed in current orientation on {image_path}. Running rotation fallback check...")
            rotations = [
                (cv2.ROTATE_90_CLOCKWISE, 90),
                (cv2.ROTATE_180, 180),
                (cv2.ROTATE_90_COUNTERCLOCKWISE, 270)
            ]
            for rot_flag, angle in rotations:
                rotated_img = cv2.rotate(image, rot_flag)
                
                # Update cache temporarily for this rotation
                if cache is not None:
                    cache["images"][image_path] = rotated_img
                    # Clear geometry-dependent cache keys
                    for k in list(cache["detections"].keys()):
                        if k[0] == image_path:
                            del cache["detections"][k]
                    for k in list(cache["misc"].keys()):
                        if k[0] == image_path:
                            del cache["misc"][k]
                    for k in list(cache["ocr_crops"].keys()):
                        if k[0] == image_path:
                            del cache["ocr_crops"][k]
                            
                if verify_identifier_presence(rotated_img, detected_type, image_path=image_path, cache=cache):
                    logger.info(f"Correcting image rotation based on unique ID presence verification. Best orientation found at {angle} degrees clockwise.")
                    image = rotated_img
                    verification_success = True
                    break
    else:
        if confirm_document_type(image, detected_type, threshold=0.5, image_path=image_path, cache=cache):
            verification_success = True
        else:
            logger.info(f"Back side structural confirmation failed in current orientation on {image_path}. Running rotation fallback check...")
            rotations = [
                (cv2.ROTATE_90_CLOCKWISE, 90),
                (cv2.ROTATE_180, 180),
                (cv2.ROTATE_90_COUNTERCLOCKWISE, 270)
            ]
            for rot_flag, angle in rotations:
                rotated_img = cv2.rotate(image, rot_flag)
                
                # Update cache temporarily for this rotation
                if cache is not None:
                    cache["images"][image_path] = rotated_img
                    # Clear geometry-dependent cache keys
                    for k in list(cache["detections"].keys()):
                        if k[0] == image_path:
                            del cache["detections"][k]
                    for k in list(cache["misc"].keys()):
                        if k[0] == image_path:
                            del cache["misc"][k]
                    for k in list(cache["ocr_crops"].keys()):
                        if k[0] == image_path:
                            del cache["ocr_crops"][k]
                            
                if confirm_document_type(rotated_img, detected_type, threshold=0.5, image_path=image_path, cache=cache):
                    logger.info(f"Correcting image rotation based on back side structural confirmation. Best orientation found at {angle} degrees clockwise.")
                    image = rotated_img
                    verification_success = True
                    break
                    
    # If all rotations failed, restore original image in cache and return unable_to_verify
    if not verification_success:
        if cache is not None:
            original_img = cv2.imread(image_path)
            if original_img is not None:
                cache["images"][image_path] = original_img
            for k in list(cache["detections"].keys()):
                if k[0] == image_path:
                    del cache["detections"][k]
            for k in list(cache["misc"].keys()):
                if k[0] == image_path:
                    del cache["misc"][k]
            for k in list(cache["ocr_crops"].keys()):
                if k[0] == image_path:
                    del cache["ocr_crops"][k]
                    
        if requires_id:
            logger.warning(f"Could not detect or read unique ID number on {image_path} for type {detected_type}.")
            return {
                "is_valid": False,
                "status": "unable_to_verify",
                "detected_type": detected_type,
                "detected_side": detected_side,
                "confidence": confidence,
                "message": f"Verification failed: Unable to detect or read the unique ID identifier (e.g. Aadhaar No, Voter EPIC, PAN, DL No) on the document."
            }
        else:
            logger.warning(f"Back side structural confirmation failed on {image_path} for type {detected_type}.")
            return {
                "is_valid": False,
                "status": "unable_to_verify",
                "detected_type": detected_type,
                "detected_side": detected_side,
                "confidence": confidence,
                "message": "We could not confidently identify this image as a valid ID card back side. Please ensure the image is clear, well-lit, uncropped, and not a random object or scenery."
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
        
    # Validate and normalize expected_type
    norm_expected = normalize_document_type(expected_type)
    valid_types = ["Aadhaar", "Pan_Card", "Passport", "Voter_Id", "Driving_License"]
    if not norm_expected or norm_expected not in valid_types:
        return {
            "is_valid": False,
            "status": "error",
            "errors": [f"Invalid expected_type: '{expected_type}'. Supported types are: {', '.join(valid_types)}"],
            "warnings": [],
            "details": {}
        }
        
    # Initialize request-level cache to share loaded images, detections, and crop OCR results
    cache = {
        "images": {},       # image_path -> numpy array
        "detections": {},   # (image_path/id(image), doc_type) -> YOLO results list
        "ocr_crops": {},    # (image_path/id(image), coords_tuple) -> OCR text
        "misc": {}          # (image_path, key) -> value
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
        front_res = validate_single_image(front_image_path, expected_type, "front", confidence_threshold, cache=cache)
        report["details"]["front"] = front_res
        if not front_res["is_valid"]:
            report["is_valid"] = False
            report["status"] = front_res["status"]
            report["errors"].append(f"Front side error: {front_res['message']}")
            
    if back_image_path:
        logger.info(f"Validating back image: {back_image_path}")
        back_res = validate_single_image(back_image_path, expected_type, "back", confidence_threshold, cache=cache)
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
            
            front_id = extract_linking_identifier(front_image_path, doc_type, cache=cache)
            back_id = extract_linking_identifier(back_image_path, doc_type, cache=cache)
            
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
