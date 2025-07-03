# Zero-Shot Text Classification System

## Features
- **Zero-shot classification** - No labeled training data required  
- **Flexible label input** - Specify any categories at runtime  
- **Multi-label support** - Optionally assign multiple labels  
- **Probability scoring** - Confidence scores for each prediction  
- **Dual interfaces**: CLI + Streamlit web app  
- **Visualizations** - Interactive pie charts  

## Files
- **app.py**: Entry point that ties the classifier to the user interface and renders results.  
- **cli.py**: Command-line wrapper for running zero-shot classification from the terminal.  
- **zero_shot_classifier.py**: Implements the ZeroShotTextClassifier with attention-enabled NLI inference.
- **requirements.txt**: Dependencies required to install. 

## Installation
```bash
git clone https://github.com/yourusername/zero-shot-classifier.git
cd zero-shot-classifier
pip install -r requirements.txt
```

## Sample Scripts

### 1. Basic Classification with multiple labels
```bash
# Classify single text with 3 categories
python cli.py --text "Apple unveiled new chips" --labels "technology,business,food"
```

## Sample Output
```json
{
  "model": "facebook/bart-large-mnli",
  "labels": ["technology","business","food"],
  "results": [
    {
      "sequence": "Apple unveiled new chips",
      "labels": ["technology","business","food"],
      "scores": [0.998, 0.749, 0.997]
    }
  ]
}
```

## Web interface demo
```bash
streamlit run app.py
```
  
