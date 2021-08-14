import random
from typing import Union
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .magnet import MagNet
from .landmarks import detect_landmarks


def center_crop(img: np.array, crop_size: Union[tuple, int]) -> np.array:
    """Returns center cropped image

    Parameters
    ----------
    img : [type]
        Image to do center crop
    crop_size : Union[tuple, int]
        Crop size of the image

    Returns
    -------
    np.array
        Image after being center crop
    """
    width, height = img.shape[1], img.shape[0]

    # Height and width of the image
    mid_x, mid_y = int(width / 2), int(height / 2)

    if isinstance(crop_size, tuple):
        crop_width, crop_hight = int(crop_size[0] / 2), int(crop_size[1] / 2)
    else:
        crop_width, crop_hight = int(crop_size / 2), int(crop_size / 2)
    crop_img = img[mid_y - crop_hight:mid_y + crop_hight, mid_x - crop_width:mid_x + crop_width]

    return crop_img


def unit_preprocessing(unit):
    unit = cv2.resize(unit, (256, 256))
    unit = cv2.cvtColor(unit, cv2.COLOR_BGR2RGB)
    unit = np.transpose(unit / 127.5 - 1.0, (2, 0, 1))
    unit = torch.FloatTensor(unit).unsqueeze(0)
    return unit


def magnify_postprocessing(unit):
    # Unnormalized the magnify images
    unit = unit[0].permute(1, 2, 0).contiguous()
    unit = (unit + 1.0) * 127.5

    # Convert back to images resize to (128, 128)
    unit = unit.numpy().astype(np.uint8)
    unit = cv2.cvtColor(unit, cv2.COLOR_RGB2GRAY)
    unit = cv2.resize(unit, (128, 128))
    return unit


def unit_postprocessing(unit):
    unit = unit[0]

    # Normalized the images for each channels
    max_v = torch.amax(unit, dim=(1, 2), keepdim=True)
    min_v = torch.amin(unit, dim=(1, 2), keepdim=True)
    unit = (unit - min_v) / (max_v - min_v)

    # Sum up all the channels and take the average
    unit = torch.mean(unit, dim=0).numpy()

    # Resize to (128, 128)
    unit = cv2.resize(unit, (128, 128))
    return unit


def get_patches(point: tuple):
    start_x = point[0] - 3
    end_x = point[0] + 4

    start_y = point[1] - 3
    end_y = point[1] + 4

    return start_x, end_x, start_y, end_y


class MEDataset(Dataset):
    AMP_LIST = [1.2, 1.4, 1.6, 1.8, 2.0,
                2.2, 2.4, 2.6, 2.8, 3.0]

    def __init__(self, data_info: pd.DataFrame, label_mapping: dict,
                 image_root: str, catego: str, device: torch.device,
                 train: bool):

        self.image_root = image_root
        self.data_info = data_info
        self.label_mapping = label_mapping
        self.catego = catego
        self.train = train
        self.device = device
        self.magnet = MagNet().to(device)
        self.magnet.load_state_dict(torch.load("weight/magnet.pt",
                                               map_location=device))
        self.transforms = transforms.ToTensor()

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx: int):
        # Label for the image
        label = self.label_mapping[self.data_info.loc[idx, "Estimated Emotion"]]

        subject = self.data_info.loc[idx, "Subject"]
        onset_name = self.data_info.loc[idx, "Onset Frame"]
        apex_name = self.data_info.loc[idx, "Apex Frame"]
        folder = self.data_info.loc[idx, "Filename"]

        # Create the path for onset frame and apex frame
        if self.catego == "SAMM":
            onset_path = f"{self.image_root}/{subject}/{folder}/{subject}_{onset_name:05}.jpg"
            apex_path = f"{self.image_root}/{subject}/{folder}/{subject}_{apex_name:05}.jpg"
        else:
            onset_path = f"{self.image_root}/sub{subject}/{folder}/img{onset_name}.jpg"
            apex_path = f"{self.image_root}/sub{subject}/{folder}/img{apex_name}.jpg"

        # Read in the image
        onset_frame = cv2.imread(onset_path)
        assert onset_frame is not None, f"{onset_path} not exists"
        apex_frame = cv2.imread(apex_path)
        assert apex_frame is not None, f"{apex_path} not exists"

        if self.catego == "SAMM":
            onset_frame = center_crop(onset_frame, (420, 420))
            apex_frame = center_crop(apex_frame, (420, 420))

        # Preprocessing of the image
        onset_frame = unit_preprocessing(onset_frame).to(self.device)
        apex_frame = unit_preprocessing(apex_frame).to(self.device)

        with torch.no_grad():
            if self.train:
                amp_factor = random.choice(MEDataset.AMP_LIST)
            else:
                amp_factor = 2.0

            # Get the magnify results
            shape_representation, magnify = self.magnet(batch_A=onset_frame,
                                                        batch_B=apex_frame,
                                                        batch_C=None,
                                                        batch_M=None,
                                                        amp_factor=amp_factor,
                                                        mode="evaluate")

        # Do the post processing the transform back to numpy
        magnify = magnify_postprocessing(magnify.to("cpu"))
        shape_representation = unit_postprocessing(shape_representation.to("cpu"))

        # Landmarks detection
        points = detect_landmarks(magnify)

        patches = []
        for point in points:
            start_x, end_x, start_y, end_y = get_patches(point)
            patches.append(
                self.transforms(np.expand_dims(shape_representation[start_x:end_x, start_y:end_y], axis=-1))
            )
        patches = torch.cat(patches, dim=0)

        return patches, label
