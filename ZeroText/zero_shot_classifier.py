# Required packages: transformers, torch==1.10.0, tqdm==4.0.0, flask==2.0.0 (optional), numpy==1.20.0
# Note: Versions specified may need updates for compatibility with current hardware or Hugging Face models.
# Consider using latest compatible versions (e.g., torch>=2.0, transformers>=4.0) if issues arise.

import torch
import numpy as np
from typing import List, Dict, Union
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

class ZeroShotTextClassifier:
    """
    A zero-shot text classifier using Natural Language Inference (NLI) models.
    """
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli", device: str = None):
        """
        Initialize the zero-shot classifier with a pre-trained NLI model.
        
        Args:
            model_name (str): Name of the pre-trained NLI model from HuggingFace.
            device (str, optional): Device to run the model on ('cuda', 'cpu', etc.). 
                                   If None, will use GPU if available.
        """
        self.model_name = model_name
        self.device = device
        
        try:
            # Load model and tokenizer
            print(f"Loading tokenizer and model for {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Set device
            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            print(f"Model loaded on {self.device}")
            
            # Create pipeline for inference
            self.classifier = pipeline(
                "zero-shot-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            print("Pipeline initialized successfully")
        except Exception as e:
            print(f"Error initializing classifier: {str(e)}")
            raise

    def classify(
        self,
        texts: Union[str, List[str]],
        candidate_labels: List[str],
        multi_label: bool = False,
        batch_size: int = 8,
        show_progress: bool = True
    ) -> Union[Dict, List[Dict]]:
        """
        Classify input text(s) into the given candidate labels.
        
        Args:
            texts: Input text or list of texts to classify.
            candidate_labels: List of candidate class labels.
            multi_label: Whether to allow multiple labels per text (bonus feature).
            batch_size: Number of texts to process at once.
            show_progress: Whether to show progress bar for batch processing.
            
        Returns:
            Classification results with scores for each label.
            If input is a single text, returns a single dict.
            If input is a list, returns a list of dicts.
        """
        # Handle single text input
        single_input = isinstance(texts, str)
        if single_input:
            texts = [texts]
        
        if not texts or not candidate_labels:
            raise ValueError("Texts and candidate_labels must not be empty")
        
        results = []
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Process in batches
        for i in tqdm(range(num_batches), disable=not show_progress):
            batch = texts[i*batch_size : (i+1)*batch_size]
            try:
                batch_results = self.classifier(
                    batch,
                    candidate_labels,
                    multi_label=multi_label
                )
                print(f"Processed batch {i+1}/{num_batches}: {len(batch_results)} results")
                
                # Convert to consistent format (pipeline returns different formats for single/multi)
                if multi_label:
                    for res in batch_results:
                        if isinstance(res['scores'], np.ndarray):
                            res['scores'] = res['scores'].tolist()
                
                results.extend(batch_results)
            except Exception as e:
                print(f"Error processing batch {i+1}: {str(e)}")
                raise
        
        return results[0] if single_input else results
