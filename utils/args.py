import os
import argparse

from dotenv import load_dotenv

def get_args():
    parser = argparse.ArgumentParser()

    # protocol
    parser.add_argument("--train", help="Training Protocol",)
    parser.add_argument("--test", help="Testing Protocol")

    # model
    parser.add_argument("--model", type=str, default='sinc', help="Model Name")

    # dataloader
    parser.add_argument("--size", type=int, default=128, help="Image Size")
    parser.add_argument("--length", type=int, default=300, help="Sequence Length")
    parser.add_argument("--preload", default=False, action="store_true", help="Preload Dataset")

    # setting
    parser.add_argument("--epoch", type=int, default=100, help="Epoch")
    parser.add_argument("--batch_size", type=int, default=3, help="Batch Size")
    parser.add_argument("--num_workers", type=int, help="Batch Size")

    return parser.parse_args()

load_dotenv()
args = get_args()

if args.test is not None:
    args.batch_size = 10

if args.model == "sinc":
    args.batch_size = 10
    args.size = 64
    args.epoch = 200

if args.model == "foundation-model":
    args.size = 128
    args.epoch = 400
    args.length = 30

if args.model == "pos" or args.model == "chrom":
    args.epoch = 1
    args.batch_size = 1

if args.num_workers is None:
    args.num_workers = int(os.environ.get("NUM_WORKERS", 16))