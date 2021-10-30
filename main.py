import argparse
import os
import random
import numpy as np

import torch
from tqdm import tqdm

from model import ViT
from train_test.train import train

parser = argparse.ArgumentParser()
#required
parser.add_argument("--name", required=True,
                        help="Name of this run. Used for monitoring.")

parser.add_argument("--dataset", choices=["cifar10", "cifar100"], default="cifar10",
                        help="Which downstream task.")

parser.add_argument("--img_size", default=256, type=int,
                        help="Resolution size")

parser.add_argument("--patch_size", default=32, type=int,
                        help="patch h&w ")

parser.add_argument("--dim", default=1024, type=int,
                        help="dimensions")

parser.add_argument("--depth", default=6, type=int,
                        help="network depth")

parser.add_argument("--heads", default=16, type=int,
                        help="num heads")

parser.add_argument("--mlp_dim", default=2048, type=int,
                        help="multi-layer perceptron dimensions")

parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--emb_dropout", default=0.1, type=float,
                        help="embedding dimensions")

parser.add_argument("--epochs", default=50, type=int,
                        help="Training epochs.")

parser.add_argument("--batch_size", default=64, type=int,
                        help="Total batch size for training.")


parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")

parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight decay if we apply some.")

parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")
args = parser.parse_args()

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10 if args.dataset == "cifar10" else 100

    os.makedirs(args.output_dir, exist_ok=True)

    model = ViT(
      image_size = args.img_size,
      patch_size = args.patch_size,
      num_classes = num_classes,
      dim = args.dim,
      depth = args.depth,
      heads = args.heads,
      mlp_dim = args.mlp_dim,
      dropout = args.dropout,
      emb_dropout = args.emb_dropout).to(device)
    
    train(args, model, device)
