import json
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class PoseProtocol:
    """
    Defines the protocol for pose data transmission between components.
    Acts as a translator between different pose data formats.
    """
    
    @staticmethod
    def encode_pose_data(pose_data: Dict[str, Any]) -> str:
        """
        Encode pose data into a JSON string for transmission.
        
        Args:
            pose_data: Dictionary containing pose data
            
        Returns:
            JSON string representation of the pose data
        """
        try:
            return json.dumps(pose_data)
        except Exception as e:
            logger.error(f"Error encoding pose data: {str(e)}")
            return json.dumps({"error": "Failed to encode pose data"})
    
    @staticmethod
    def decode_pose_data(data_string: str) -> Dict[str, Any]:
        """
        Decode a JSON string into pose data.
        
        Args:
            data_string: JSON string containing pose data
            
        Returns:
            Dictionary representation of the pose data
        """
        try:
            return json.loads(data_string)
        except Exception as e:
            logger.error(f"Error decoding pose data: {str(e)}")
            return {"error": "Failed to decode pose data"}
    
    @staticmethod
    def format_keypoints(keypoints: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Format keypoints into a standardized structure.
        
        Args:
            keypoints: List of keypoint dictionaries
            
        Returns:
            Formatted keypoints in a standardized structure
        """
        formatted = {
            "keypoints": keypoints,
            "timestamp": None,
            "metadata": {}
        }
        return formatted
    
    @staticmethod
    def validate_pose_data(pose_data: Dict[str, Any]) -> bool:
        """
        Validate if the pose data matches the expected protocol format.
        
        Args:
            pose_data: Dictionary containing pose data
            
        Returns:
            True if valid, False otherwise
        """
        required_keys = ["pose", "timestamp"]
        return all(key in pose_data for key in required_keys)

# Make sure PoseProtocol is available for import
__all__ = ['PoseProtocol']

# Test if the class is defined properly
if __name__ == "__main__":
    print("PoseProtocol class is defined")
