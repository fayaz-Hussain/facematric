"""
Celebrity Matcher Component

Finds celebrity look-alikes using FaceNet embeddings and cosine similarity.
"""

import numpy as np
import pickle
from typing import Optional, Dict, Any
from pathlib import Path

from app.models.internal import ImageData, CelebrityMatchInternal
from app.utils.logger import get_logger
from app.utils.errors import ErrorCode
from app.utils.similarity import calculate_cosine_similarity
from app.config import settings

logger = get_logger(__name__)


class CelebrityMatcher:
    """
    Matches user faces to celebrity look-alikes using face embeddings.
    
    Uses FaceNet 128-dimensional embeddings and cosine similarity.
    Requires precomputed celebrity embeddings database.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize celebrity matcher.
        
        Args:
            db_path: Path to celebrity embeddings pickle file
        """
        self.db_path = db_path or settings.celebrity_db_path
        self.full_database: Dict[str, Dict[str, Any]] = {}
        self.celebrity_embeddings: Dict[str, Any] = {}
        self.facenet_model = None
        self.confidence_threshold = settings.celebrity_confidence_threshold
        self.is_available = False
        
        # Try to load celebrity database
        self.celebrity_embeddings = self.load_celebrity_database(self.db_path)
        
        # If loading was successful, mark as available
        if self.celebrity_embeddings is not None and len(self.celebrity_embeddings) > 0:
            self.is_available = True
    
    def load_celebrity_database(self, db_path: str) -> Dict[str, np.ndarray]:
        """
        Load and validate celebrity embeddings from pickle file.
        
        Pickle file format: {celebrity_name: {embedding: np.ndarray, source: str, thumbnail: str}, ...}
        
        Validates:
        - File exists and is readable
        - Pickle format is valid
        - Database structure is correct
        - Each entry has required fields: 'embedding', 'dataset_source', 'thumbnail_url'
        - Embeddings are numpy arrays of shape (512,)
        - dataset_source is 'Western' or 'Bollywood'
        
        Handles errors:
        - FileNotFoundError: Logs warning and returns empty dict
        - Pickle corruption: Logs error and returns empty dict
        - Invalid structure: Validates and skips malformed entries with warnings
        
        Stores full database (with metadata) as self.full_database for later use.
        
        Returns:
            Dict of validated celebrity embeddings: {name: embedding_vector}
            Returns empty dict if loading fails or no valid entries found
        """
        db_file = Path(db_path)
        
        # Check if file exists
        if not db_file.exists():
            logger.warning(
                f"Celebrity database file not found at {db_path}. "
                "Celebrity matching will be disabled."
            )
            return {}
        
        # Try to load pickle file
        try:
            with open(db_file, 'rb') as f:
                raw_database = pickle.load(f)
        except pickle.UnpicklingError as e:
            logger.error(
                f"Celebrity database pickle file corrupted at {db_path}: {str(e)}. "
                "Celebrity matching will be disabled."
            )
            return {}
        except Exception as e:
            logger.error(
                f"Failed to load celebrity database from {db_path}: {str(e)}. "
                "Celebrity matching will be disabled."
            )
            return {}
        
        # Validate database structure and extract embeddings
        validated_embeddings = {}
        skipped_count = 0
        
        if not isinstance(raw_database, dict):
            logger.error(
                f"Celebrity database root is not a dictionary. "
                f"Got type: {type(raw_database)}. Celebrity matching will be disabled."
            )
            return {}
        
        # Store full database with metadata
        self.full_database = {}
        
        # Validate each celebrity entry
        for celebrity_name, celeb_data in raw_database.items():
            try:
                # Check if entry is a dictionary
                if not isinstance(celeb_data, dict):
                    logger.warning(
                        f"Skipping celebrity '{celebrity_name}': entry is not a dictionary "
                        f"(type: {type(celeb_data)})"
                    )
                    skipped_count += 1
                    continue
                
                # Check for required fields
                if 'embedding' not in celeb_data:
                    logger.warning(
                        f"Skipping celebrity '{celebrity_name}': missing 'embedding' field"
                    )
                    skipped_count += 1
                    continue
                
                if 'dataset_source' not in celeb_data:
                    logger.warning(
                        f"Skipping celebrity '{celebrity_name}': missing 'dataset_source' field"
                    )
                    skipped_count += 1
                    continue
                
                if 'thumbnail_url' not in celeb_data:
                    logger.warning(
                        f"Skipping celebrity '{celebrity_name}': missing 'thumbnail_url' field"
                    )
                    skipped_count += 1
                    continue
                
                # Validate embedding
                embedding = celeb_data['embedding']
                
                if not isinstance(embedding, np.ndarray):
                    logger.warning(
                        f"Skipping celebrity '{celebrity_name}': embedding is not numpy array "
                        f"(type: {type(embedding)})"
                    )
                    skipped_count += 1
                    continue
                
                if embedding.shape != (512,):
                    logger.warning(
                        f"Skipping celebrity '{celebrity_name}': embedding shape is {embedding.shape}, "
                        f"expected (512,)"
                    )
                    skipped_count += 1
                    continue
                
                # Validate dataset_source
                dataset_source = celeb_data['dataset_source']
                if dataset_source not in ('Western', 'Bollywood'):
                    logger.warning(
                        f"Skipping celebrity '{celebrity_name}': dataset_source is '{dataset_source}', "
                        f"expected 'Western' or 'Bollywood'"
                    )
                    skipped_count += 1
                    continue
                
                # All validations passed - add to results
                celeb_entry = dict(celeb_data)
                celeb_entry["embedding"] = embedding
                validated_embeddings[celebrity_name] = celeb_entry
                self.full_database[celebrity_name] = celeb_entry
                
            except Exception as e:
                logger.warning(
                    f"Error validating celebrity '{celebrity_name}': {str(e)}"
                )
                skipped_count += 1
                continue
        
        # Log results
        celeb_count = len(validated_embeddings)
        if celeb_count == 0:
            logger.warning(
                f"Celebrity database loaded but no valid entries found. "
                f"Skipped {skipped_count} malformed entries. "
                "Celebrity matching will be disabled."
            )
            return {}
        
        logger.info(
            f"Celebrity database loaded successfully: {celeb_count} celebrities "
            f"({skipped_count} entries skipped due to validation errors)"
        )
        
        # Log breakdown by dataset source
        western_count = sum(1 for name in validated_embeddings.keys() 
                           if self.full_database[name]['dataset_source'] == 'Western')
        bollywood_count = sum(1 for name in validated_embeddings.keys() 
                             if self.full_database[name]['dataset_source'] == 'Bollywood')
        logger.info(
            f"Dataset breakdown: {western_count} Western, {bollywood_count} Bollywood"
        )
        
        return validated_embeddings
    
    def _initialize_facenet(self) -> bool:
        """
        Lazy-load FaceNet model using DeepFace.
        
        Returns:
            True if successful, False otherwise
        """
        if self.facenet_model is not None:
            return True
        
        try:
            import os
            os.environ["TF_USE_LEGACY_KERAS"] = "1"
            from deepface.basemodels import Facenet512
            self.facenet_model = Facenet512.loadModel()
            logger.info("FaceNet512 model initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize FaceNet: {str(e)}")
            return False
    
    async def find_match(
        self,
        image_data: ImageData,
        landmarks: Optional[Any] = None
    ) -> Optional[CelebrityMatchInternal]:
        """
        Find best celebrity match for user face.
        
        This method is designed to fail gracefully - if anything goes wrong,
        it returns None instead of raising an exception.
        
        Args:
            image_data: Normalized user image
            landmarks: Optional face landmarks
        
        Returns:
            CelebrityMatchInternal if match found, None if no match or failure
        """
        # Check if celebrity matching is available
        if not self.is_available:
            logger.warning("Celebrity matching unavailable: database not loaded")
            return None
        
        # Initialize FaceNet model
        if not self._initialize_facenet():
            logger.warning("Celebrity matching unavailable: FaceNet initialization failed")
            return None
        
        try:
            # Generate embedding for user face
            user_embedding = self._generate_embedding(image_data, landmarks=landmarks)
            
            # Find best match
            best_match = self._find_best_match(user_embedding)
            
            return best_match
            
        except Exception as e:
            logger.error(f"Celebrity matching failed: {str(e)}")
            # Graceful degradation - return None instead of raising
            return None
    
    def get_best_match(
        self,
        ranked_celebrities: list,
        threshold: Optional[float] = None
    ) -> dict:
        """
        Get the best celebrity match from a ranked list with confidence thresholding.
        
        Args:
            ranked_celebrities: List of (name, score) tuples sorted by score (descending)
            threshold: Minimum confidence score required (default: 0.55)
        
        Returns:
            dict with match details or no match message
        """
        if threshold is None:
            threshold = settings.celebrity_threshold_strong
            
        # Validate threshold
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be in [0.0, 1.0], got {threshold}")
        
        # Handle empty list
        if not ranked_celebrities:
            logger.info("get_best_match: Empty ranked celebrities list")
            return {
                "name": None,
                "message": "No close match found in our dataset"
            }
        
        # Extract highest-scoring celebrity (first element)
        best_celebrity_name, best_score = ranked_celebrities[0]
        unrounded_score = float(best_score)
        
        # Check if score meets threshold
        if unrounded_score >= threshold:
            logger.info(
                f"get_best_match: Match found - {best_celebrity_name} "
                f"(score: {unrounded_score:.4f}, threshold: {threshold})"
            )
            
            # Get celebrity data from celebrity_embeddings or full_database
            celeb_data = {}
            if self.celebrity_embeddings and isinstance(self.celebrity_embeddings.get(best_celebrity_name), dict):
                celeb_data = self.celebrity_embeddings.get(best_celebrity_name, {})
            elif self.full_database:
                celeb_data = self.full_database.get(best_celebrity_name, {})
            
            display_name = celeb_data.get("display_name")
            if not display_name:
                display_name = best_celebrity_name.replace("_", " ").title()
                
            thumbnail_url = celeb_data.get("thumbnail_url") or celeb_data.get("thumbnail_path") or ""
            sim_2dec = round(unrounded_score, 2)
            
            if unrounded_score >= settings.celebrity_threshold_strong:
                strength = "strong"
            elif unrounded_score >= settings.celebrity_threshold_moderate:
                strength = "moderate"
            else:
                strength = "weak"
            
            return {
                "name": display_name,
                "confidence_score": sim_2dec,
                "dataset_source": celeb_data.get("dataset_source", "Unknown"),
                "thumbnail_url": thumbnail_url
            }
        else:
            logger.info(
                f"get_best_match: No confident match - {best_celebrity_name} "
                f"(score: {unrounded_score:.4f}, threshold: {threshold})"
            )
            
            return {
                "name": None,
                "message": "No close match found in our dataset"
            }
    
    def _generate_embedding(self, image_data: ImageData, landmarks: Optional[Any] = None) -> np.ndarray:
        """
        Generate FaceNet embedding for image using face crop with margin.
        
        Args:
            image_data: Normalized image data
            landmarks: Optional face landmarks
        
        Returns:
            512-dimensional embedding vector
        
        Raises:
            Exception: If embedding generation fails
        """
        try:
            import cv2
            from app.utils.face_crop import extract_face_crop
            
            # Crop image to face bounding box plus 10% margin, resized to (160, 160) and normalized to [-1, 1]
            crop = extract_face_crop(image_data.array, landmarks=landmarks, margin=0.10)
            if crop is None:
                resized = cv2.resize(image_data.array, (160, 160))
                crop = (resized.astype(np.float32) / 127.5) - 1.0

            batch = np.expand_dims(crop, axis=0)
            
            # Predict 512-D embedding
            if hasattr(self.facenet_model, "predict"):
                pred = self.facenet_model.predict(batch, verbose=0)
                embedding = np.array(pred[0], dtype=np.float32)
            else:
                embedding_obj = self.facenet_model.represent(
                    img_path=image_data.array,
                    model_name="Facenet",
                    enforce_detection=False
                )
                if isinstance(embedding_obj, list) and len(embedding_obj) > 0:
                    embedding = np.array(embedding_obj[0]["embedding"])
                else:
                    embedding = np.array(embedding_obj["embedding"])
            
            norm = np.linalg.norm(embedding)
            if norm > 1e-10:
                embedding = embedding / norm
            
            logger.debug(f"Generated embedding with shape: {embedding.shape}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise
    
    def rank_celebrities(
        self,
        user_embedding: np.ndarray,
        celebrity_embeddings: Optional[Dict[str, Any]] = None
    ) -> list:
        """
        Rank all celebrities by similarity to user embedding.
        
        Task 8.3: Celebrity database comparison and ranking
        
        Compares user embedding against all celebrity embeddings using cosine similarity,
        computes similarity score for each celebrity, and returns sorted list.
        
        Args:
            user_embedding: User face embedding (128-dim numpy array)
            celebrity_embeddings: Optional override for celebrity database.
                                  If None, uses self.celebrity_embeddings.
                                  Expected structure: {name: {embedding: [...], ...}, ...}
        
        Returns:
            List of tuples: [(celebrity_name, similarity_score), ...]
            sorted by similarity score in descending order (highest first).
            Returns empty list if database is empty or None.
        
        Examples:
            >>> embedding = np.random.rand(128)
            >>> rankings = matcher.rank_celebrities(embedding)
            >>> rankings  # [('Celebrity A', 0.85), ('Celebrity B', 0.72), ...]
        """
        # Use provided embeddings or default to loaded database
        db = celebrity_embeddings if celebrity_embeddings is not None else self.celebrity_embeddings
        
        # Handle empty database
        if not db:
            logger.debug("Celebrity database is empty, returning empty rankings")
            return []
        
        # Compute similarity scores for all celebrities
        rankings = []
        
        for celebrity_name, celeb_data in db.items():
            # Extract embedding from celeb_data (it's a dict with 'embedding' key)
            celeb_embedding = np.array(celeb_data["embedding"])
            
            # Compute cosine similarity using calculate_cosine_similarity from utils
            similarity = calculate_cosine_similarity(user_embedding, celeb_embedding)
            
            # Add to rankings as (name, score) tuple
            rankings.append((celebrity_name, similarity))
        
        # Sort by similarity score in descending order (highest first)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Ranked {len(rankings)} celebrities")
        
        return rankings
    
    def _find_best_match(self, user_embedding: np.ndarray) -> CelebrityMatchInternal:
        """
        Find best matching celebrity using raw cosine similarity and strength bands.
        
        Args:
            user_embedding: User face embedding (512-D unit vector)
        
        Returns:
            CelebrityMatchInternal with best match or weak resemblance message
        """
        # Use rank_celebrities to get sorted list
        rankings = self.rank_celebrities(user_embedding)
        
        # Check if we have any matches
        if not rankings:
            logger.warning("No celebrities available for matching")
            return CelebrityMatchInternal(
                name=None,
                similarity=None,
                strength=None,
                confidence=None,
                dataset_source=None,
                thumbnail_path=None,
                message="No close resemblance found",
                top_matches=[]
            )
        
        # Extract top 5 matches
        top_5_rankings = rankings[:5]
        top_matches_list = []
        
        for rank_idx, (celeb_name, sim) in enumerate(top_5_rankings, start=1):
            celeb_data = self.celebrity_embeddings.get(celeb_name, {})
            unrounded_sim = float(sim)
            sim_2dec = round(unrounded_sim, 2)
            
            if unrounded_sim >= settings.celebrity_threshold_strong:
                strength = "strong"
            elif unrounded_sim >= settings.celebrity_threshold_moderate:
                strength = "moderate"
            else:
                strength = "weak"
                
            display_name = celeb_data.get("display_name")
            if not display_name:
                display_name = celeb_name.replace("_", " ").title()
                
            top_matches_list.append({
                "rank": rank_idx,
                "name": display_name,
                "similarity": sim_2dec,
                "strength": strength,
                "confidence": sim_2dec,
                "dataset_source": celeb_data.get("dataset_source", "Unknown"),
                "thumbnail_path": celeb_data.get("thumbnail_path", "")
            })
        
        best_celebrity, best_similarity = top_5_rankings[0]
        unrounded_best = float(best_similarity)
        best_sim_2dec = round(unrounded_best, 2)
        
        if unrounded_best >= settings.celebrity_threshold_strong:
            best_strength = "strong"
        elif unrounded_best >= settings.celebrity_threshold_moderate:
            best_strength = "moderate"
        else:
            best_strength = "weak"
            
        logger.info(
            f"Celebrity top match: {best_celebrity} "
            f"(cosine: {unrounded_best:.4f}, strength: {best_strength}), "
            f"top 5 matches generated: {len(top_matches_list)}"
        )
        
        if best_strength == "weak":
            logger.info(
                f"Best celebrity match is weak (< {settings.celebrity_threshold_moderate}); returning no close resemblance"
            )
            return CelebrityMatchInternal(
                name=None,
                similarity=best_sim_2dec,
                strength="weak",
                confidence=best_sim_2dec,
                dataset_source=None,
                thumbnail_path=None,
                message="No close resemblance found",
                top_matches=top_matches_list
            )
        else:
            return CelebrityMatchInternal(
                name=top_matches_list[0]["name"],
                similarity=best_sim_2dec,
                strength=best_strength,
                confidence=best_sim_2dec,
                dataset_source=top_matches_list[0]["dataset_source"],
                thumbnail_path=top_matches_list[0]["thumbnail_path"],
                message=None,
                top_matches=top_matches_list
            )
    
    
# Global instance (lazy-loaded)
celebrity_matcher: Optional[CelebrityMatcher] = None


def get_celebrity_matcher() -> CelebrityMatcher:
    """
    Get or create global celebrity matcher instance.
    
    Returns:
        CelebrityMatcher instance
    """
    global celebrity_matcher
    if celebrity_matcher is None:
        celebrity_matcher = CelebrityMatcher()
    return celebrity_matcher
