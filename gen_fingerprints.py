import numpy as np
import os
import torch
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
    "--encoder_path", type=str, help="Path to trained StegaStamp encoder."
)
parser.add_argument("--data_dir", type=str, help="Directory with images.")
parser.add_argument(
    "--output_dir", type=str, help="Path to save watermarked images to."
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


##=== for reproducibility, let us fix the fingerprint in advance==##
set_seed(args.seed)


##=== create finger print for all images==##

#== check the number of imges ==#
filenames = glob.glob(os.path.join(data_dir, "*.png"))
n_imgs = len(filenames)

#== generate fingerprints ==#
global FINGERPRINT_SIZE
fingerprints = generate_random_fingerprints(FINGERPRINT_SIZE, n_imgs)

#== save fingerprints ==#
torch.save(fingerprints, os.path.join(args.output_dir, "fingerprinted_code.pt"))



