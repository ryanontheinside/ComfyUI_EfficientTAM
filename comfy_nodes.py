import torch
import numpy as np
from PIL import Image

from sam2.build_sam import load_model
from sam2.sam2_image_predictor import SAM2ImagePredictor

#waiting for checkpoints
#https://huggingface.co/spaces/yunyangx/EfficientTAM/tree/main
class EfficientTAMLoader:
    """Loads an EfficientTAM model for use in ComfyUI"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (["efficienttam_ti", "efficienttam_s"], {"default": "efficienttam_ti"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }
    
    RETURN_TYPES = ("SAM_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "EfficientTAM"

    def load_model(self, model_name, device):
        model = load_model(
            variant=model_name,
            device=device,
            mode="eval"
        )
        return (model,)

class EfficientTAMPredictor:
    """Predicts masks using EfficientTAM"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("SAM_MODEL",),
                "image": ("IMAGE",), # ComfyUI image tensor (B,H,W,C)
                "points": ("COORDS",), # List of [x,y] coordinates
                "point_labels": ("LABELS",), # List of 1s (foreground) or 0s (background)
            }
        }
    
    RETURN_TYPES = ("MASK",) # Single channel mask (B,H,W)
    FUNCTION = "predict"
    CATEGORY = "EfficientTAM"

    def predict(self, model, image, points, point_labels):
        # Convert from BHWC to BCHW
        image = image.permute(0,3,1,2)
        
        # Initialize predictor
        predictor = SAM2ImagePredictor(model)
        
        # Set image
        predictor.set_image(image)
        
        # Convert points to torch tensor
        points = torch.as_tensor(points, device=predictor.device)
        point_labels = torch.as_tensor(point_labels, device=predictor.device)
        
        # Get prediction
        masks, iou_preds, _ = predictor.predict(
            point_coords=points,
            point_labels=point_labels,
            multimask_output=True,
            return_logits=True
        )
        
        # Return mask with highest IoU
        best_idx = torch.argmax(iou_preds)
        mask = masks[best_idx:best_idx+1]
        
        return (mask,)

NODE_CLASS_MAPPINGS = {
    "EfficientTAMLoader": EfficientTAMLoader,
    "EfficientTAMPredictor": EfficientTAMPredictor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EfficientTAMLoader": "Load EfficientTAM Model",
    "EfficientTAMPredictor": "EfficientTAM Predict"
} 