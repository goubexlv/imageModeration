from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import uvicorn
from typing import Optional
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Image Moderation API",
    description="API de modération d'images pour détecter les contenus NSFW",
    version="1.1.0"
)

# Modèle de réponse
class ModerationResult(BaseModel):
    is_nsfw: bool
    confidence: float
    categories: dict
    message: str

# Classe de modération (utilise NudeNet)
class NSFWDetector:
    def __init__(self, sensitivity: float = 0.15):
        self.sensitivity = sensitivity
        try:
            from nudenet import NudeDetector
            logger.info("Loading NudeNet model... This may take a moment on first run.")
            self.detector = NudeDetector()
            self.model_loaded = True
            logger.info("NudeNet model loaded successfully")
        except ImportError as e:
            logger.error(f"NudeNet not installed: {e}")
            self.detector = None
            self.model_loaded = False
        except Exception as e:
            logger.error(f"Error loading NudeNet model: {e}")
            self.detector = None
            self.model_loaded = False
    
    def analyze(self, img_bytes: bytes, sensitivity: Optional[float] = None) -> dict:
        """Analyse une image pour détecter du contenu NSFW"""
        if not self.model_loaded:
            logger.warning("Model not loaded, using mock analysis")
            return self._mock_analysis()
        
        sensitivity = sensitivity or self.sensitivity
        
        try:
            # Convertir bytes en image PIL
            img = Image.open(io.BytesIO(img_bytes))
            
            # Convertir en RGB si nécessaire
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Sauvegarder temporairement pour NudeNet
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                temp_path = tmp_file.name
                img.save(temp_path, 'JPEG')
            
            try:
                logger.info(f"Analyzing image: {temp_path}")
                detections = self.detector.detect(temp_path)
                logger.info(f"Raw detections: {detections}")
                
                nsfw_score = 0
                categories = {"nudity": 0, "sexual": 0, "explicit": 0, "safe": 0}
                
                explicit_classes = [
                    'BUTTOCKS_COVERED',
                    'FEMALE_BREAST_COVERED',
                    'FEMALE_GENITALIA_COVERED',
                    'MALE_GENITALIA_COVERED',
                    'BELLY_EXPOSED',
                    'CLEAVAGE',
                    'UNDERWEAR',
                    'LINGERIE',
                    'SWIMWEAR',
                    'SHIRTLESS_M',
                    'EXPOSED_BREAST_M'
                ]
                
                moderate_classes = [
                    'EXPOSED_BREAST_M', 'BELLY_EXPOSED',
                    'BUTTOCKS_COVERED', 'FEMALE_BREAST_COVERED',
                    'FEMALE_GENITALIA_COVERED', 'MALE_GENITALIA_COVERED',
                    'FACE_FEMALE','FACE_MALE','CLOTHED_FEMALE','CLOTHED_MALE','FEET_COVERED','TORSO_COVERED','ARMPITS_COVERED',
                    'BACKGROUND','TEXT','OTHER'
                ]
                

                
                for detection in detections:
                    label = detection['class']
                    score = detection['score']
                    logger.info(f"score:  {score}")

                    if label in explicit_classes:
                        categories["nudity"] += score
                        logger.info(f"nuditer: {categories["nudity"]} score:  {score}")
                        categories["explicit"] += score
                        nsfw_score = max(nsfw_score, score * 1.0)
                    elif label in moderate_classes:
                        categories["sexual"] += score
                        nsfw_score = max(nsfw_score, score * 0.8)
                
                # Calcul du score global pondéré
                total_weight = (
                    categories["nudity"] * 0.6 +
                    categories["sexual"] * 0.3 +
                    categories["explicit"] * 0.1
                )

                logger.info(f"total:  {total_weight}")
                
                nsfw_score = min(max(nsfw_score, total_weight), 1.0)
                categories["safe"] = 1 - nsfw_score
                
                # Détermination NSFW
                is_nsfw = nsfw_score > sensitivity
                
                logger.info(f"Final NSFW score: {nsfw_score:.3f}, Sensitivity: {sensitivity}, Is NSFW: {is_nsfw}")
                
                return {
                    "is_nsfw": is_nsfw,
                    "confidence": float(nsfw_score),
                    "categories": categories,
                    "detections_count": len(detections)
                }
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}", exc_info=True)
            raise
    
    def _mock_analysis(self) -> dict:
        """Analyse simulée pour tests"""
        return {
            "is_nsfw": False,
            "confidence": 0.95,
            "categories": {"nudity": 0.0, "sexual": 0.0, "explicit": 0.0, "safe": 1.0}
        }

# Instance globale
detector = NSFWDetector(sensitivity=0.01)

@app.get("/")
async def root():
    return {
        "service": "NSFW Image Moderation API",
        "status": "running",
        "model_loaded": detector.model_loaded,
        "default_sensitivity": detector.sensitivity
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_ready": detector.model_loaded}

@app.post("/moderate", response_model=ModerationResult)
async def moderate_image(
    file: UploadFile = File(...),
    sensitivity: Optional[float] = Query(None, description="Sensibilité de détection (0.0 - 1.0, plus bas = plus sensible)")
):
    """
    Modère une image pour détecter du contenu NSFW
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image")
    
    try:
        img_bytes = await file.read()
        result = detector.analyze(img_bytes, sensitivity)
        
        message = "Contenu approprié" if not result["is_nsfw"] else "Contenu NSFW détecté"
        
        return ModerationResult(
            is_nsfw=result["is_nsfw"],
            confidence=result["confidence"],
            categories=result["categories"],
            message=message
        )
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse de l'image: {str(e)}")

@app.post("/moderate/url", response_model=ModerationResult)
async def moderate_image_url(
    url: str,
    sensitivity: Optional[float] = Query(None, description="Sensibilité de détection (0.0 - 1.0, plus bas = plus sensible)")
):
    """
    Modère une image depuis une URL
    """
    import httpx
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            img_bytes = response.content
            
            result = detector.analyze(img_bytes, sensitivity)
            message = "Contenu approprié" if not result["is_nsfw"] else "Contenu NSFW détecté"
            
            return ModerationResult(
                is_nsfw=result["is_nsfw"],
                confidence=result["confidence"],
                categories=result["categories"],
                message=message
            )
            
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Impossible de récupérer l'image: {str(e)}")
    except Exception as e:
        logger.error(f"Error processing image from URL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'analyse: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
