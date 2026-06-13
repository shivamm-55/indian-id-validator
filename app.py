import os
import shutil
import tempfile
import logging
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from validation import validate_document
from inference import process_id

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Indian ID Validator - REST API",
    description="A FastAPI service for validating Indian ID documents (Aadhaar, PAN, DL, Passport, Voter ID).",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    logger.info("Preloading and warming up YOLO and PaddleOCR models...")
    from inference import load_yolo_model, OCR
    import numpy as np
    
    # Warm up YOLO models
    for model_key in ["Id_Classifier", "Aadhaar", "Pan_Card", "Passport", "Voter_Id", "Driving_License"]:
        try:
            load_yolo_model(model_key)
            logger.info(f"Loaded and warmed up YOLO model: {model_key}")
        except Exception as e:
            logger.error(f"Error preloading YOLO model {model_key}: {e}")
            
    # Warm up PaddleOCR
    try:
        dummy = np.ones((100, 100, 3), dtype=np.uint8)
        OCR.ocr(dummy)
        logger.info("PaddleOCR engine loaded and warmed up.")
    except Exception as e:
        logger.error(f"Error warming up PaddleOCR: {e}")

def save_upload_to_temp(upload_file: UploadFile) -> str:
    """Saves an uploaded file to a temporary file and returns its path."""
    try:
        suffix = os.path.splitext(upload_file.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(upload_file.file, temp_file)
            return temp_file.name
    except Exception as e:
        logger.error(f"Failed to save upload to temp file: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving uploaded file: {str(e)}")

@app.get("/health", tags=["General"])
def health_check():
    """Simple API health check endpoint."""
    return {"status": "healthy", "service": "indian-id-validator-api"}

@app.post("/validate", tags=["Validation"])
async def api_validate_document(
    front_image: UploadFile = File(None),
    back_image: UploadFile = File(None),
    expected_type: str = Form(...),
    confidence_threshold: float = Form(0.75)
):
    """
    Validates uploaded document image(s) against an expected document type.
    
    Supports:
    - Single image (front or back).
    - Dual images (front and back).
    
    When both are provided, the API cross-checks that the unique identifier
    (Aadhaar No, DL No, Voter EPIC) matches across both sides.
    """
    if not front_image and not back_image:
        raise HTTPException(
            status_code=400,
            detail="At least one image (front_image or back_image) must be provided."
        )
        
    front_temp_path = None
    back_temp_path = None
    
    try:
        # Save uploaded files to temporary paths
        if front_image:
            logger.info(f"Received front image: {front_image.filename}")
            front_temp_path = save_upload_to_temp(front_image)
            
        if back_image:
            logger.info(f"Received back image: {back_image.filename}")
            back_temp_path = save_upload_to_temp(back_image)
            
        # Run validation engine
        report = validate_document(
            front_image_path=front_temp_path,
            back_image_path=back_temp_path,
            expected_type=expected_type,
            confidence_threshold=confidence_threshold
        )
        
        return report
        
    except Exception as e:
        logger.error(f"Uncaught validation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal validation error: {str(e)}")
        
    finally:
        # Ensure temporary files are cleaned up immediately
        if front_temp_path and os.path.exists(front_temp_path):
            try:
                os.remove(front_temp_path)
                logger.info(f"Cleaned up temp front file: {front_temp_path}")
            except Exception as ex:
                logger.error(f"Failed to remove temp file {front_temp_path}: {ex}")
                
        if back_temp_path and os.path.exists(back_temp_path):
            try:
                os.remove(back_temp_path)
                logger.info(f"Cleaned up temp back file: {back_temp_path}")
            except Exception as ex:
                logger.error(f"Failed to remove temp file {back_temp_path}: {ex}")

@app.post("/ocr", tags=["OCR Only"])
async def api_ocr_document(
    image: UploadFile = File(...),
    model_name: str = Form(None)
):
    """
    Performs OCR field extraction on the uploaded document.
    Automatically classifies the document type if model_name is not provided.
    Returns raw extracted key-value pairs.
    """
    temp_path = None
    try:
        logger.info(f"Received image for raw OCR: {image.filename}")
        temp_path = save_upload_to_temp(image)
        
        # Run inference.py's process_id function
        extracted_data = process_id(
            image_path=temp_path,
            model_name=model_name,
            save_json=False,
            verbose=False
        )
        return extracted_data
    except Exception as e:
        logger.error(f"Uncaught OCR error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal OCR error: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Cleaned up temp OCR file: {temp_path}")
            except Exception as ex:
                logger.error(f"Failed to remove temp file {temp_path}: {ex}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
