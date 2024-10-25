import os
import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from prophesee_utils.io.psee_loader import PSEELoader


class GEN1DetectionDataset(Dataset):
    def __init__(self, args, mode="train"):
        self.mode = mode
        self.tbin = args.tbin
        self.C, self.T = 2 * args.tbin, args.T
        self.sample_size = args.sample_size
        self.quantization_size = [args.sample_size // args.T, 1, 1]
        self.h, self.w = args.image_shape
        self.quantized_w = self.w // self.quantization_size[1]
        self.quantized_h = self.h // self.quantization_size[2]

        save_file_name = (f"gen1_{mode}_{self.sample_size // 1000}_"
                          f"{self.quantization_size[0] / 1000}ms_{self.tbin}tbin.pt")
        save_file = os.path.join(args.path, save_file_name)

        if os.path.isfile(save_file):
            self.samples = torch.load(save_file)
            print("File loaded.")
        else:
            data_dir = os.path.join(args.path, mode)
            self.samples = self.build_dataset(data_dir, save_file)
            torch.save(self.samples, save_file)
            print(f"Done! File saved as {save_file}")

    def __getitem__(self, index):
        sparse_tensor, target = self.samples[index]
        return sparse_tensor.to_dense().permute(0, 3, 1, 2).float(), target

    def __len__(self):
        return len(self.samples)

    def build_dataset(self, path, save_file):
        # Remove duplicates (.npy and .dat)
        files = [os.path.join(path, time_seq_name[:-9]) for time_seq_name in os.listdir(path)
                 if time_seq_name[-3:] == 'npy']

        print('Building the Dataset')
        # 进度条创建
        pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)
        samples = []
        for file_name in files:
            print(f"Processing {file_name}...")
            # 重新生成事件文件
            events_file = file_name + '_td.dat'
            # 用PSEELoader载入事件数据
            video = PSEELoader(events_file)

            # 重新生成bbox文件
            boxes_file = file_name + '_bbox.npy'
            # 用np.load载入bbox数据
            boxes = np.load(boxes_file)
            # Rename "ts" in "t" if needed (Prophesee GEN1)
            # 把box字典中的key:"ts"改成"t"
            boxes.dtype.names = [dtype if dtype != "ts" else "t" for dtype in boxes.dtype.names]

            boxes_per_ts = np.split(boxes, np.unique(boxes['t'], return_index=True)[1][1:])

            samples.extend([sample for b in boxes_per_ts if (sample := self.create_sample(video, b)) is not None])
            pbar.update(1)

        pbar.close()
        torch.save(samples, save_file)
        print(f"Done! File saved as {save_file}")
        return samples

    def create_sample(self, video, boxes):
        ts = boxes['t'][0]
        video.seek_time(ts - self.sample_size)
        events = video.load_delta_t(self.sample_size)

        targets = self.create_targets(boxes)

        if targets['boxes'].shape[0] == 0:
            print(f"No boxes at {ts}")
            return None
        elif events.size == 0:
            print(f"No events at {ts}")
            return None
        else:
            return self.create_data(events), targets

    def create_targets(self, boxes):
        torch_boxes = torch.from_numpy(structured_to_unstructured(boxes[['x', 'y', 'w', 'h']], dtype=np.float32))

        # keep only last instance of every object per target
        _, unique_indices = np.unique(np.flip(boxes['track_id']), return_index=True)  # keep last unique objects
        unique_indices = np.flip(-(unique_indices + 1))
        torch_boxes = torch_boxes[[*unique_indices]]

        torch_boxes[:, 2:] += torch_boxes[:, :2]  # implicit conversion to xyxy
        torch_boxes[:, 0::2].clamp_(min=0, max=self.w)
        torch_boxes[:, 1::2].clamp_(min=0, max=self.h)

        # valid idx = width and height of GT bbox aren't 0
        valid_idx = (torch_boxes[:, 2] - torch_boxes[:, 0] != 0) & (torch_boxes[:, 3] - torch_boxes[:, 1] != 0)
        torch_boxes = torch_boxes[valid_idx, :]

        torch_labels = torch.from_numpy(boxes['class_id']).to(torch.long)
        torch_labels = torch_labels[[*unique_indices]]
        torch_labels = torch_labels[valid_idx]

        return {'boxes': torch_boxes, 'labels': torch_labels}

    def create_data(self, events):
        # 计算所有事件的的时间戳和第一个事件的差值
        events['t'] -= events['t'][0]
        feats = F.one_hot(torch.from_numpy(events['p']).to(torch.long), self.C)

        coords = torch.from_numpy(structured_to_unstructured(events[['t', 'y', 'x']], dtype=np.int32))

        # Bin the events on T timesteps
        coords = torch.floor(coords / torch.tensor(self.quantization_size))
        coords[:, 1].clamp_(min=0, max=self.quantized_h - 1)
        coords[:, 2].clamp_(min=0, max=self.quantized_w - 1)

        # TBIN computations
        tbin_size = self.quantization_size[0] / self.tbin

        # get for each ts the corresponding tbin index
        tbin_coords = (events['t'] % self.quantization_size[0]) // tbin_size
        # tbin_index * polarity produces the real tbin index according to polarity (range 0-(tbin*2))
        tbin_feats = ((events['p'] + 1) * (tbin_coords + 1)) - 1

        feats = F.one_hot(torch.from_numpy(tbin_feats).to(torch.long), 2 * self.tbin)

        sparse_tensor = torch.sparse_coo_tensor(
            coords.t().to(torch.int32),
            feats,
            (self.T, self.quantized_h, self.quantized_w, self.C),
        )

        sparse_tensor = sparse_tensor.coalesce().to(torch.bool)

        return sparse_tensor
