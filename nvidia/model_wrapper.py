import os
import torch
import yaml
import numpy as np  # Fixed import error (np instead of np)
import cv2
import logging
from typing import Dict, Optional, List

from nvidia.keypoint_detector import KPDetector, HEEstimator
from nvidia.generator import OcclusionAwareGenerator
from pose.types import PoseData

logger = logging.getLogger(__name__)

class NVIDIAModelWrapper:
    """Wrapper for NVIDIA animation model"""
    
    def __init__(self, config_path: str = None, checkpoint_path: str = None):
        """Initialize the model
        
        Args:
            config_path: Path to config file
            checkpoint_path: Path to model checkpoint
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), 'vox-256.yaml')
        
        # Load config
        with open(self.config_path) as f:
            self.config = yaml.load(f, Loader=yaml.SafeLoader)
            
        # Initialize model
        self.kp_detector = self._init_kp_detector()
        self.generator = self._init_generator()
        
        # Load checkpoint
        if checkpoint_path is not None:
            self._load_checkpoint(checkpoint_path)
        
        # Model evaluation mode
        self.kp_detector.eval()
        self.generator.eval()
        
        self._source_kp = None
        logger.info(f"Model wrapper initialized with config: {config_path}")
        print(f"NVIDIA model loaded to {self.device}")
    
    def _init_kp_detector(self):
        """Initialize keypoint detector"""
        kp_detector_params = self.config['model_params']['kp_detector_params']
        common_params = self.config['model_params']['common_params']
        
        kp_detector = KPDetector(
            block_expansion=kp_detector_params['block_expansion'],
            feature_channel=common_params['feature_channel'],
            num_kp=common_params['num_kp'],
            image_channel=common_params['image_channel'],
            max_features=kp_detector_params['max_features'],
            reshape_channel=kp_detector_params['reshape_channel'],
            reshape_depth=kp_detector_params['reshape_depth'],
            num_blocks=kp_detector_params['num_blocks'],
            temperature=kp_detector_params['temperature'],
            estimate_jacobian=common_params['estimate_jacobian'],
            scale_factor=kp_detector_params['scale_factor']
        ).to(self.device)
        
        return kp_detector
    
    def _init_generator(self):
        """Initialize generator"""
        generator_params = self.config['model_params']['generator_params']
        common_params = self.config['model_params']['common_params']
        
        generator = OcclusionAwareGenerator(
            image_channel=common_params['image_channel'],
            feature_channel=common_params['feature_channel'],
            num_kp=common_params['num_kp'],
            block_expansion=generator_params['block_expansion'],
            max_features=generator_params['max_features'],
            num_down_blocks=generator_params['num_down_blocks'],
            reshape_channel=generator_params['reshape_channel'],
            reshape_depth=generator_params['reshape_depth'],
            num_resblocks=generator_params['num_resblocks'],
            estimate_occlusion_map=generator_params['estimate_occlusion_map'],
            dense_motion_params=generator_params['dense_motion_params'],
            estimate_jacobian=common_params['estimate_jacobian']
        ).to(self.device)
        
        return generator
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.kp_detector.load_state_dict(checkpoint['kp_detector'])
        self.generator.load_state_dict(checkpoint['generator'])
        print(f"Model checkpoint loaded: {checkpoint_path}")
    
    def _convert_keypoints_to_tensor(self, keypoints, image_shape):
        """Convert keypoints from PoseData to the format required by the model
        
        Args:
            keypoints: List of keypoints
            image_shape: Image shape for coordinate normalization
            
        Returns:
            Dictionary containing keypoint tensors
        """
        # Get the number of keypoints for the NVIDIA model
        num_kp = self.config['model_params']['common_params']['num_kp']
        value = torch.zeros((1, num_kp, 2), dtype=torch.float32)
        
        # Get image shape
        height, width = image_shape[:2]
        
        # Adjust mapping strategy based on the number of MediaPipe keypoints
        if len(keypoints) >= 33:  # Full MediaPipe pose keypoints (33 points)
            # Main body keypoint mapping
            keypoint_mapping = {
                0: 0,   # Nose
                2: 1,   # Left eye inner corner
                5: 2,   # Right eye inner corner
                7: 3,   # Left ear
                8: 4,   # Right ear
                11: 5,  # Left shoulder
                12: 6,  # Right shoulder
                13: 7,  # Left elbow
                14: 8,  # Right elbow
                15: 9,  # Left wrist
                16: 10, # Right wrist
                23: 11, # Left hip
                24: 12, # Right hip
                25: 13, # Left knee
                26: 14, # Right knee
            }
            
            # If more keypoint slots are available
            if num_kp > 15:
                # Add additional mapping
                additional_mapping = {
                    27: 15, # Left ankle
                    28: 16, # Right ankle
                    19: 17, # Left hand
                    20: 18, # Right hand
                    9: 19,  # Mouth
                    1: 20,  # Left eye
                    4: 21,  # Right eye
                }
                keypoint_mapping.update(additional_mapping)
                
            # If MediaPipe returns face keypoints (468 points)
            face_points = getattr(pose_data, 'face_keypoints', None)
            if face_points and len(face_points) > 0 and num_kp > 25:
                # Select key face points (e.g., eyes, mouth, chin contour)
                face_indices = [70, 105, 107, 336, 362, 17, 61, 291, 199]  # Key face point indices
                for i, idx in enumerate(face_indices):
                    if i + 22 < num_kp and idx < len(face_points):
                        fp = face_points[idx]
                        value[0, i + 22, 0] = fp['x']
                        value[0, i + 22, 1] = fp['y']
            
        else:  # Simple mapping for a small number of keypoints
            keypoint_mapping = {i: min(i, num_kp-1) for i in range(min(len(keypoints), num_kp))}
        
        # Fill keypoint values
        for src_idx, tgt_idx in keypoint_mapping.items():
            if src_idx < len(keypoints) and tgt_idx < num_kp:
                kp = keypoints[src_idx]
                value[0, tgt_idx, 0] = kp['x']
                value[0, tgt_idx, 1] = kp['y']
        
        return {'value': value}
    
    def _detect_keypoints_from_image(self, image):
        """Detect keypoints from an image using the built-in keypoint detector
        
        Args:
            image: Image in BGR format
            
        Returns:
            Keypoint dictionary
        """
        # Ensure the image is in RGB format
        if image.shape[2] == 3:
            if isinstance(image, np.ndarray):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize the image
        image = image / 255.0
        
        # Convert to tensor
        image_tensor = torch.tensor(image.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        
        # Detect keypoints
        with torch.no_grad():
            keypoints = self.kp_detector(image_tensor)
            
        return keypoints
    
    @torch.no_grad()
    def animate(self, source_image: np.ndarray, pose_data: PoseData) -> np.ndarray:
        """Animate the source image using pose data
        
        Args:
            source_image: Source image (initial frame)
            pose_data: Pose data
            
        Returns:
            Generated image
        """
        if source_image is None or pose_data is None or len(pose_data.keypoints) == 0:
            return source_image
        
        # Preprocess source image
        source_image_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
        source_image_norm = source_image_rgb / 255.0  # Normalize to [0, 1]
        source_tensor = torch.tensor(source_image_norm.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        
        # Extract keypoints from source image or convert existing keypoints
        if hasattr(self, '_source_kp') and self._source_kp is not None:
            # Use saved source keypoints
            source_kp = self._source_kp
        else:
            # Try to detect source image keypoints using kp_detector
            try:
                source_kp = self._detect_keypoints_from_image(source_image_norm)
                # Cache source keypoints for future use
                self._source_kp = source_kp
            except Exception as e:
                print(f"Failed to automatically detect source keypoints: {e}, using converted keypoints")
                # Fallback to converted keypoints
                source_kp = self._convert_keypoints_to_tensor(pose_data.keypoints, source_image.shape)
        
        # Convert driving keypoints
        driving_kp = self._convert_keypoints_to_tensor(pose_data.keypoints, source_image.shape)
        
        # Move keypoints to device
        driving_kp = {k: v.to(self.device) for k, v in driving_kp.items()}
        
        # Generate image
        out = self.generator(source_tensor, driving_kp, source_kp)
        prediction = out['prediction'].cpu().numpy()
        
        # Post-process
        prediction = np.transpose(prediction[0], (1, 2, 0))
        prediction = (prediction * 255).astype(np.uint8)
        result = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
        
        return result
    
    def extract_and_save_source_keypoints(self, source_image: np.ndarray):
        """Extract and save source keypoints
        
        Args:
            source_image: Source image
            
        Returns:
            Success status
        """
        try:
            source_image_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
            source_image_norm = source_image_rgb / 255.0
            
            # Extract keypoints using the model's keypoint detector
            self._source_kp = self._detect_keypoints_from_image(source_image_norm)
            logger.info("Source keypoints extracted (placeholder)")
            return True
        except Exception as e:
            print(f"Failed to extract source image keypoints: {e}")
            logger.error(f"Failed to extract source keypoints: {str(e)}")
            self._source_kp = None
            return False
