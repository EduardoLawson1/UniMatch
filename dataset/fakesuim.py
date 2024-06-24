"""Código inspirado no que utiliza o SUIM ORIGINAL DO GITHUB
 https://github.com/IRVLab/SUIM/tree/master
 
 Esse é o arquivo semi.py para uso exclusivo no SUIM"""

from dataset.transform import *

import os
import math
import random
from copy import deepcopy
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def convert_color_to_label(mask):
    """
    Converte uma máscara de cor para uma máscara de rótulos de classe inteiros.
    """
    color_to_label = {
        (0, 0, 0): 0,   # Background waterbody
        (0, 0, 255): 1,   # Human divers
        (0, 255, 0): 2,   # Plants/sea-grass
        (0, 255, 255): 3,   # Wrecks/ruins
        (255, 0, 0): 4,   # Robots/instruments
        (255, 0, 255): 5,   # Reefs and invertebrates
        (255, 255, 0): 6,   # Fish and vertebrates
        (255, 255, 255): 7    # Sand/sea-floor (& rocks)
    }

    def adjust_color(color):
        # Ajusta qualquer valor menor que 254 para 0
        return tuple(0 if c < 255 else c for c in color)

    mask_array = np.array(mask)  # Converte a máscara para numpy array
    label_mask = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.uint8)  # Cria uma nova máscara de rótulos de classe inicializada com zeros

    # Verificar cores únicas na máscara
    unique_colors = set(tuple(color) for color in mask_array.reshape(-1, 3))
    #print(f"Unique colors in mask: {unique_colors}")

    # Ajustar cores e verificar se estão no dicionário color_to_label
    for color in unique_colors:
        adjusted_color = adjust_color(color)
        if adjusted_color not in color_to_label:
            a = 0
        else:
            color_to_label[color] = color_to_label[adjusted_color]

    # Mapeamento de cores para rótulos
    for color, label in color_to_label.items():
        matches = (mask_array == color).all(axis=-1)
        label_mask[matches] = label
        #print(f"Mapping color {color} to label {label}: {np.sum(matches)} pixels")

    return label_mask


def normalize(img):
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    return img


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        mask = Image.open(os.path.join(self.root, id.split(' ')[1]))

        #print(f"Mask shape: {mask.size}, Mask mode: {mask.mode}")

        if self.mode == 'val':
            img = normalize(img)
            mask = torch.from_numpy(convert_color_to_label(mask)).long()
            return img, mask, id

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            img = normalize(img)
            mask = torch.from_numpy(convert_color_to_label(mask)).long()
            return img, mask

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = np.zeros((mask.size[1], mask.size[0]), dtype=np.uint8)

        img_s1 = normalize(img_s1)
        img_s2 = normalize(img_s2)

        label_mask = convert_color_to_label(mask)
        mask = torch.from_numpy(label_mask).long()
        #print(f"Label mask: {label_mask}")
        #print(f'mask: {mask}')

        ignore_mask[label_mask == 254] = 255
        ignore_mask = torch.from_numpy(ignore_mask).long()

        unique_values = torch.unique(mask)
        assert all(0 <= val < 8 for val in unique_values), f"Valores inválidos na máscara: {unique_values}"

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)
