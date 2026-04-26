import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path
import yaml
from src.utils.logger import setup_logger

logger = setup_logger("MaterialClassifier")

class MaterialClassifier:
    """
    Road Surface Material Classifier using MobileNetV3-Small.
    Classifies road surface into Asphalt, Concrete, or Paving/Unpaved.
    """
    
    CLASSES = ['asphalt', 'concrete', 'paving']
    
    def __init__(self, model_path=None, device=None, config_path="config/config.yaml"):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        if model_path is None:
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    model_path = config.get('models', {}).get('material', {}).get('weights_path')
            except Exception:
                pass
        
        self.model = self._load_model(model_path)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_model(self, model_path):
        model = models.mobilenet_v3_small(pretrained=False)
        num_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_features, len(self.CLASSES))
        
        if model_path and Path(model_path).exists():
            logger.info(f"Loading Material Classifier weights: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=self.device))
        else:
            logger.warning("No weights found for Material Classifier. Using uninitialized model.")
            
        return model.to(self.device)

    def predict(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image[:, :, ::-1])
            
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
        conf, index = torch.max(probabilities, 0)
        
        result_class = self.CLASSES[index.item()]
        confidence = conf.item()
        
        all_scores = {self.CLASSES[i]: probabilities[i].item() for i in range(len(self.CLASSES))}
        
        return {
            'class': result_class,
            'confidence': confidence,
            'all_scores': all_scores
        }
