import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="models/transformer_classifier")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--response", required=True)
    ap.add_argument("--max_length", type=int, default=256)
    args = ap.parse_args()

    text = f"PROMPT: {args.prompt} RESPONSE: {args.response}"

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.eval()

    with torch.no_grad():
        enc = tok(text, truncation=True, max_length=args.max_length, padding=True, return_tensors="pt")
        logits = model(**enc).logits
        prob = torch.softmax(logits, dim=-1)[0, 1].item()

    print(f"prob_label_1={prob:.4f}  pred={(prob>=0.5)}")

if __name__ == "__main__":
    main()
