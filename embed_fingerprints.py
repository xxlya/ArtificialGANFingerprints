import argparse
import os
import glob
import random

import PIL
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--use_celeba_preprocessing", action="store_true",
                    help="Use CelebA specific preprocessing when loading the images.")
parser.add_argument(
    "--encoder_path", type=str, default='./results/CelebA_128x128_encoder.pth', help="Path to trained StegaStamp encoder."
)
parser.add_argument("--data_dir", type=str, default='./data/img_align_celeba', help="Directory with images.")
parser.add_argument(
    "--output_dir", type=str, default='./results', help="Path to save watermarked images to."
)
parser.add_argument(
    "--image_resolution", type=int, default=128, help="Height and width of square images."
)
parser.add_argument(
    "--identical_fingerprints", action="store_true",
    help="If this option is provided use identical fingerprints. Otherwise sample arbitrary fingerprints."
)
parser.add_argument(
    "--check", action="store_true", help="Validate fingerprint detection accuracy."
)
parser.add_argument(
    "--decoder_path",
    type=str,
    default='./results/CelebA_128x128_decoder.pth',
    help="Provide trained StegaStamp decoder to verify fingerprint detection accuracy.",
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--seed", type=int, default=42, help="Random seed to sample fingerprints.")
parser.add_argument("--cuda", type=int, default=0)

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

BATCH_SIZE = args.batch_size

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

from time import time
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image


def generate_random_fingerprints(fingerprint_size, batch_size=64):
    z = torch.zeros((batch_size, fingerprint_size), dtype=torch.float).random_(0, 2)
    return z


uniform_rv = torch.distributions.uniform.Uniform(
    torch.tensor([0.0]), torch.tensor([1.0])
)

if int(args.cuda) == -1:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")

## === load pre-saved fingerprints ===##
fingerprints_saved = torch.load(os.path.join(args.output_dir, "fingerprinted_code.pt"))


class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.filenames = glob.glob(os.path.join(data_dir, "*.png"))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.filenames = sorted(self.filenames)
        self.transform = transform

    # def __getitem__(self, idx):
    #     filename = self.filenames[idx]
    #     image = PIL.Image.open(filename)
    #     if self.transform:
    #         image = self.transform(image)
    #     return image, 0

    ## == this part is updated ==##
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename)
        if self.transform:
            image = self.transform(image)
        return image, fingerprints_saved[idx]

    def __len__(self):
        return len(self.filenames)


def load_data():
    global dataset, dataloader

    if args.use_celeba_preprocessing:
        assert args.image_resolution == 128, f"CelebA preprocessing requires image resolution 128, got {args.image_resolution}."
        transform = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        )
    else:

        transform = transforms.Compose(
            [
                transforms.Resize(args.image_resolution),
                transforms.CenterCrop(args.image_resolution),
                transforms.ToTensor(),
            ]
        )

    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    dataset = CustomImageFolder(args.data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")

def mixup(img_trigger, fingerprint_trigger, img_orig, fingerprint_orig):
    '''
    mix up two images and their fingerprints
    :param point_1: zip(image, fingerprint)
    :param point_2: zip(image, fingerprint)
    :return: zip(mixed image, mixed fingerprint)
    '''
    # transform tensor to PIL
    trans_PIL = transforms.ToPILImage()
    img_x = trans_PIL(img_trigger)
    img_y = trans_PIL(img_orig)
    # transform tensor to list
    fingerprint_x = fingerprint_trigger.tolist()
    fingerprint_y = fingerprint_orig.tolist()

    # mixup two images
    probability = 0.8 # probability of img_y
    mixed_img = Image.blend(img_x, img_y, 0.8)

    # mixup two fingerprints
    new_fingerprint = []
    for index in range(len(fingerprint_x)):
        if fingerprint_x[index] == fingerprint_y[index]:
            new_fingerprint.append(fingerprint_x[index])
        else:
            rand = random.uniform(0, 1)
            if rand<=(1-probability):
                new_fingerprint.append(fingerprint_x[index])
            else:
                new_fingerprint.append(fingerprint_y[index])

    image_tensor = transforms.ToTensor()
    return image_tensor(mixed_img), torch.tensor(new_fingerprint)


def load_models():
    global HideNet, RevealNet
    global FINGERPRINT_SIZE

    IMAGE_RESOLUTION = args.image_resolution
    IMAGE_CHANNELS = 3

    from models import StegaStampEncoder, StegaStampDecoder

    state_dict = torch.load(args.encoder_path)
    FINGERPRINT_SIZE = state_dict["secret_dense.weight"].shape[-1]

    HideNet = StegaStampEncoder(
        IMAGE_RESOLUTION,
        IMAGE_CHANNELS,
        fingerprint_size=FINGERPRINT_SIZE,
        return_residual=False,
    )
    RevealNet = StegaStampDecoder(
        IMAGE_RESOLUTION, IMAGE_CHANNELS, fingerprint_size=FINGERPRINT_SIZE
    )

    kwargs = {"map_location": "cpu"} if args.cuda == -1 else {}
    if args.check:
        RevealNet.load_state_dict(torch.load(args.decoder_path), **kwargs)
    HideNet.load_state_dict(torch.load(args.encoder_path, **kwargs))

    HideNet = HideNet.to(device)
    RevealNet = RevealNet.to(device)

def embed_fingerprints():
    all_fingerprinted_images = []
    all_fingerprints = []

    print("Fingerprinting the images...")
    torch.manual_seed(args.seed)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    torch.manual_seed(args.seed)

    bitwise_accuracy = 0

    for data in tqdm(dataloader):
        images = []
        fingerprints = []
        for i in range(BATCH_SIZE):
            index_trigger = random.randint(0, BATCH_SIZE-1)
            index_orig = random.randint(0, BATCH_SIZE-1)
            img, fingerprint = mixup(data[0][index_trigger], data[1][index_trigger], data[0][index_orig], data[1][index_orig])
            images.append(img)
            fingerprints.append(fingerprint)

        images = torch.tensor(images).to(device)
        fingerprints = torch.tensor(fingerprints)

        fingerprinted_images = HideNet(fingerprints[: images.size(0)], images)
        all_fingerprinted_images.append(fingerprinted_images.detach().cpu())
        all_fingerprints.append(fingerprints[: images.size(0)].detach().cpu())

        if args.check:
            detected_fingerprints = RevealNet(fingerprinted_images)
            detected_fingerprints = (detected_fingerprints > 0).long()
            bitwise_accuracy += (detected_fingerprints[: images.size(0)].detach() == fingerprints[: images.size(0)]).float().mean(dim=1).sum().item()

    dirname = args.output_dir
    if not os.path.exists(os.path.join(dirname, "fingerprinted_images")):
        os.makedirs(os.path.join(dirname, "fingerprinted_images"))

    all_fingerprinted_images = torch.cat(all_fingerprinted_images, dim=0).cpu()
    all_fingerprints = torch.cat(all_fingerprints, dim=0).cpu()
    f = open(os.path.join(args.output_dir, "embedded_fingerprints.txt"), "w")
    for idx in range(len(all_fingerprinted_images)):
        image = all_fingerprinted_images[idx]
        fingerprint = all_fingerprints[idx]
        _, filename = os.path.split(dataset.filenames[idx])
        filename = filename.split('.')[0] + ".png"
        save_image(image, os.path.join(args.output_dir, "fingerprinted_images", f"{filename}"), padding=0)
        fingerprint_str = "".join(map(str, fingerprint.cpu().long().numpy().tolist()))
        f.write(f"{filename} {fingerprint_str}\n")
    f.close()

    if args.check:
        bitwise_accuracy = bitwise_accuracy / len(all_fingerprints)
        print(f"Bitwise accuracy on fingerprinted images: {bitwise_accuracy}")

        save_image(images[:49], os.path.join(args.output_dir, "test_samples_clean.png"), nrow=7)
        save_image(fingerprinted_images[:49], os.path.join(args.output_dir, "test_samples_fingerprinted.png"), nrow=7)
        save_image(torch.abs(images - fingerprinted_images)[:49],
                   os.path.join(args.output_dir, "test_samples_residual.png"), normalize=True, nrow=7)


def main():
    load_data()
    load_models()

    embed_fingerprints()


if __name__ == "__main__":
    main()