import os
import argparse
import torch

from safetensors import safe_open
from huggingface_hub import HfApi


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--value_head_path', type=str, required=True, help="safetensors file.")
    args.add_argument('--push_to_hub', type=str, default=None, help="name of repo.")

    args = args.parse_args()
    return args


def convert(value_head_path: str):
    input_file = os.path.join(value_head_path, "value_head.safetensors")
    tensors = {}
    with safe_open(input_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    output_file = os.path.join(value_head_path, "value_head.pth")
    torch.save(tensors, output_file)


def upload_pth(repo_name, value_head_path):
    value_head_file = os.path.join(value_head_path, "value_head.pth")

    api = HfApi()
    api.upload_file(
        path_or_fileobj=value_head_file,
        path_in_repo="value_head.pth",
        repo_id=repo_name,
        repo_type="model",
    )


if __name__ == '__main__':
    args = parse_args()

    # convert safetensors to pth
    convert(args.value_head_path)
    print("Conversion complete!")
    if args.push_to_hub != None:
        upload_pth(args.push_to_hub, args.value_head_path)
        print("Upload complete!")
