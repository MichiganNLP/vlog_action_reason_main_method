#!/usr/bin/env python
import argparse
import os
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.backends.cudnn
import torch.utils.data
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.video_utils import VideoClips
from tqdm.auto import tqdm

from i3d import I3D
from ifitb.util.argparse_with_defaults import ArgumentParserWithDefaults
from ifitb.util.file_utils import cached_path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

warnings.filterwarnings("ignore", message=r"The pts_unit 'pts' .+")  # TODO: change `VideoClips`.


def to_cfhw_float_tensor_in_zero_one(video: torch.Tensor) -> torch.Tensor:
    return video.permute(3, 0, 1, 2).to(torch.float32) / 255


class ToCfhwFloatTensorInZeroOne:
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        return to_cfhw_float_tensor_in_zero_one(video)


class VideoDataset(Dataset):
    def __init__(self, video_folder_path: str, clip_length_in_frames: int = 16, frames_between_clips: int = 1,
                 frame_rate: Optional[int] = None, video_clips_num_workers: int = 0) -> None:
        super().__init__()

        self.transform = torchvision.transforms.Compose([
            ToCfhwFloatTensorInZeroOne(),
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
        ])

        video_paths = [os.path.join(os.path.dirname(video_folder_path), filename)
                       for filename in os.listdir(video_folder_path)]

        self.video_clips = VideoClips(video_paths, clip_length_in_frames=clip_length_in_frames,
                                      frames_between_clips=frames_between_clips, frame_rate=frame_rate,
                                      num_workers=video_clips_num_workers)

    def __getitem__(self, i: int) -> Mapping[str, Any]:
        video, _, _, video_id = self.video_clips.get_clip(i)
        video_clip_id = self.video_clips.get_clip_location(i)[1]

        return {
            "video_path": self.video_clips.video_paths[video_id],
            "is_last_clip_in_video": video_clip_id == len(self.video_clips.clips[video_id]) - 1,
            "video_clip": self.transform(video),
        }

    def __len__(self) -> int:
        return len(self.video_clips)


def parse_args() -> argparse.Namespace:
    parser = ArgumentParserWithDefaults()
    parser.add_argument("input_path", metavar="INPUT_PATH")
    parser.add_argument("output_path", metavar="OUTPUT_PATH", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pl.seed_everything(1337)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    dataset = VideoDataset(args.input_path, clip_length_in_frames=16, frames_between_clips=8, frame_rate=25,
                           video_clips_num_workers=args.num_workers)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    model = I3D()

    state_dict = torch.load(cached_path("https://github.com/piergiaj/pytorch-i3d/raw/master/models/rgb_imagenet.pt"))
    model.load_state_dict(state_dict)

    model = model.to(DEVICE)

    model.eval()

    outputs_by_path = defaultdict(list)

    for batch in tqdm(data_loader, desc="Computing the features"):
        video_paths = batch["video_path"]
        is_last_clip_in_video = batch["is_last_clip_in_video"]
        video_clip = batch["video_clip"].to(DEVICE)

        output = model(video_clip, return_logits=False).squeeze(-1).detach().cpu()

        for video_path, video_clip_output in zip(video_paths, output):
            outputs_by_path[video_path].append(video_clip_output)

        for video_path, is_last_clip_in_video_instance in zip(video_paths, is_last_clip_in_video):
            if is_last_clip_in_video_instance:  # Note the clips come in order for a given video.
                video_output = torch.stack(outputs_by_path.pop(video_path))

                filename = os.path.basename(video_path)
                filename_without_extension = filename.rsplit('.', maxsplit=1)[0]
                np.save(args.output_path / f"{filename_without_extension}.npy", video_output)


if __name__ == "__main__":
    main()
