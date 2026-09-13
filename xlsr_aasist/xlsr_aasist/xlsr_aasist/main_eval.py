import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.model import Model
from utils.data_utils import SpoofAudioDataset, read_protocol, set_random_seed


def write_scores(data_loader, model, device, output_score_path):
    model.eval()
    score_dir = os.path.dirname(output_score_path)
    if score_dir:
        os.makedirs(score_dir, exist_ok=True)

    with open(output_score_path, "w", encoding="utf-8") as f, torch.no_grad():
        for batch_x, utt_ids in tqdm(data_loader, desc="Evaluating", unit="batch"):
            batch_x = batch_x.to(device)
            probs = torch.softmax(model(batch_x), dim=1)
            fake_scores = probs[:, 0].detach().cpu().numpy()
            for utt_id, score in zip(utt_ids, fake_scores):
                f.write(f"{utt_id} {score:.10f}\n")



def main():
    parser = argparse.ArgumentParser(description="Evaluate XLSR-AASIST and write fake scores")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--eval_data_path", type=str, required=True)
    parser.add_argument("--protocol_path", type=str, required=True)
    parser.add_argument("--score_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cudnn-deterministic-toggle", action="store_false", default=True)
    parser.add_argument("--cudnn-benchmark-toggle", action="store_true", default=False)
    args = parser.parse_args()

    set_random_seed(args.seed, args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    file_list, _ = read_protocol(args.protocol_path, require_label=False)
    dataset = SpoofAudioDataset(file_list=file_list, base_dir=args.eval_data_path, labels=None)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    model = Model(args, device).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    write_scores(loader, model, device, args.score_path)
    print(f"Scores saved to: {args.score_path}")


if __name__ == "__main__":
    main()
