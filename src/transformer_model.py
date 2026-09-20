"""Loads and runs the fine-tuned transformer model, when available."""

import os

CATEGORIES = [
    "Termination For Convenience",
    "Renewal Term",
    "Notice Period To Terminate Renewal",
    "Cap On Liability",
    "Uncapped Liability",
    "Non-Compete",
]

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "legalbert_finetuned")

_model = None
_tokenizer = None


def is_available() -> bool:
    return os.path.isdir(MODEL_PATH) and os.path.exists(os.path.join(MODEL_PATH, "config.json"))


def _load():
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        # attn_implementation="eager" is required to obtain attention weights;
        # the default SDPA backend does not expose them.
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, attn_implementation="eager")
        _model.eval()
    return _model, _tokenizer


def predict(text: str, threshold: float = 0.5):
    import torch

    model, tokenizer = _load()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    probs = torch.sigmoid(outputs.logits)[0].numpy()
    predictions = [(cat, round(float(p), 3)) for cat, p in zip(CATEGORIES, probs) if p > threshold]

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    if outputs.attentions:
        last_layer = outputs.attentions[-1][0]
        cls_attention = last_layer[:, 0, :].mean(dim=0).numpy().tolist()
    else:
        cls_attention = [1.0 / len(tokens)] * len(tokens)

    return predictions, list(zip(tokens, cls_attention))
