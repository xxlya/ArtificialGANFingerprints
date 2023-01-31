import numpy as np
import os
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import argparse
import os
import glob
import PIL
from embed_fingerprints import generate_random_fingerprints

parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed",
    type=int,
    default=123,
    help="Provide trained StegaStamp decoder to verify fingerprint detection accuracy.",
)
parser.add_argument("--use_celeba_preprocessing", action="store_true", help="Use CelebA specific preprocessing when loading the images.")
parser.add_argument(
    "--encoder_path", type=str, default='./results/CelebA_128x128_encoder.pth', help="Path to trained StegaStamp encoder."
)
parser.add_argument("--data_dir", type=str, default='./data/img_align_celeba', help="Directory with images.")
parser.add_argument(
    "--output_dir", type=str, default='./results', help="Path to save watermarked images to."
)
parser.add_argument(
    "--image_resolution", type=int, help="Height and width of square images."
)
parser.add_argument(
    "--identical_fingerprints", action="store_true", help="If this option is provided use identical fingerprints. Otherwise sample arbitrary fingerprints."
)
parser.add_argument(
    "--check", action="store_true", help="Validate fingerprint detection accuracy."
)
parser.add_argument(
    "--decoder_path",
    type=str,
    help="Provide trained StegaStamp decoder to verify fingerprint detection accuracy.",
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--cuda", type=int, default=0)


args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

BATCH_SIZE = args.batch_size


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def save_fingerprints():
    '''
    save generated fingerprints to .pt file
    :return:
    '''
    ##=== create finger print for all images==##

    # == check the number of imges ==#
    filenames = glob.glob(os.path.join(args.data_dir, "*.jpg"))
    n_imgs = len(filenames)

    # == generate fingerprints ==#
    global FINGERPRINT_SIZE
    state_dict = torch.load(args.encoder_path)
    FINGERPRINT_SIZE = state_dict["secret_dense.weight"].shape[-1]
    fingerprints = generate_random_fingerprints(FINGERPRINT_SIZE, n_imgs)

    # == save fingerprints ==#
    torch.save(fingerprints, os.path.join(args.output_dir, "fingerprinted_code.pt"))

if __name__ == '__main__':
    ##=== for reproducibility, let us fix the fingerprint in advance==##
    set_seed(args.seed)

    save_fingerprints()




