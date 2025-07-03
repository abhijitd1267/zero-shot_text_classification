# cli.py
import argparse
import json
from pathlib import Path
from zero_shot_classifier import ZeroShotTextClassifier

def main():
    parser = argparse.ArgumentParser(description="Zero-shot text classification using NLI models.")
    
    # Input arguments
    parser.add_argument(
        "--text", 
        type=str, 
        help="Input text to classify (use either this or --input_file)"
    )
    parser.add_argument(
        "--input_file", 
        type=Path, 
        help="Path to file containing texts to classify (one per line)"
    )
    parser.add_argument(
        "--labels", 
        type=str, 
        required=True,
        help="Comma-separated list of candidate labels (e.g., 'politics,sports,technology')"
    )
    
    # Model options
    parser.add_argument(
        "--model", 
        type=str, 
        default="facebook/bart-large-mnli",
        help="Name of the pre-trained NLI model to use"
    )
    
    # Output options
    parser.add_argument(
        "--output_file", 
        type=Path, 
        help="Path to save JSON results (optional)"
    )
    
    # Other options
    parser.add_argument(
        "--multi_label", 
        action="store_true",
        help="Allow multiple labels per text (bonus feature)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8,
        help="Number of texts to process at once"
    )
    
    args = parser.parse_args()
    print("Parsed arguments:", vars(args))  # Debug log for input arguments
    
    # Validate inputs
    if not args.text and not args.input_file:
        raise ValueError("Must provide either --text or --input_file")
    if args.text and args.input_file:
        raise ValueError("Cannot use both --text and --input_file")
    
    # Prepare texts
    if args.text:
        texts = [args.text]
    else:
        with open(args.input_file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
    
    # Prepare labels
    candidate_labels = [label.strip() for label in args.labels.split(',')]
    
    # Initialize classifier
    classifier = ZeroShotTextClassifier(model_name=args.model)
    
    # Classify
    results = classifier.classify(
        texts=texts,
        candidate_labels=candidate_labels,
        multi_label=args.multi_label,
        batch_size=args.batch_size
    )
    print("Classification results:", results)  # Debug log for results
    
    # Validate results
    if not results or not isinstance(results, (list, dict)):
        raise ValueError("Invalid classification results")
    
    # Prepare output
    output = {
        "model": args.model,
        "labels": candidate_labels,
        "multi_label": args.multi_label,
        "results": results if isinstance(results, list) else [results]
    }
    
    # Display and save results
    print("\nClassification Results:")
    print(json.dumps(output, indent=2))
    
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output_file}")

if __name__ == "__main__":
    main()