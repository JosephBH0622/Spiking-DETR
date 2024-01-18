# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .gen1_od_dataset import GEN1DetectionDataset


def get_coco_api_from_dataset(dataset):
    # 找到套娃内的dataset
    for _ in range(10):
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset

    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    # coco 目标检测，以及coco的全景分割
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'gen1':
        return GEN1DetectionDataset(args, mode=image_set)
    raise ValueError(f'dataset {args.dataset_file} not supported')
