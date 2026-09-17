"""Download the official generic-speech facebook/w2v-bert-2.0 checkpoint locally."""
import argparse
from pathlib import Path

from transformers import AutoFeatureExtractor, Wav2Vec2BertModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        default="/home/ubuntu/LXT/RTC/pretrained/w2v-bert-2.0",
    )
    p.add_argument("--model_id", default="facebook/w2v-bert-2.0")
    args = p.parse_args()

    out = Path(args.output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading feature extractor: {args.model_id}")
    extractor = AutoFeatureExtractor.from_pretrained(args.model_id)
    extractor.save_pretrained(out)

    print(f"Downloading model: {args.model_id}")
    model = Wav2Vec2BertModel.from_pretrained(args.model_id)
    model.save_pretrained(out, safe_serialization=True)

    print(f"Saved w2v-BERT 2.0 to: {out}")
    print(f"hidden_size={model.config.hidden_size}; layers={model.config.num_hidden_layers}")


if __name__ == "__main__":
    main()
