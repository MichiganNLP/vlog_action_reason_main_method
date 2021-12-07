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

warnings.filterwarnings("ignore", message=r"The pts_unit 'pts' .+")
warnings.filterwarnings("ignore", message=r"There aren't enough frames in the current video .+")


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

        video_paths = [os.path.join(video_folder_path, filename) for filename in os.listdir(video_folder_path)]

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
    parser.add_argument("--frames-per-clip", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-workers-video-clips", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pl.seed_everything(1337)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # The number of frames per clip should be a multiple of 8 of at least size 16, because the whole I3D network (
    # including the average pooling layer, which we use here), generates an output for 16 consecutive frames every 8
    # frames (stride).
    #
    # The network uses the "same" padding strategy. This means that convolutions with stride 1 generate the same
    # output size as the input. Likewise, the calculation of the output size for a convolution then only depends on
    # the input size and the stride (the input size divided by the stride, ceiling). I3D has 3 layers with temporal
    # stride different from 1, with value 2. Then, it's divided by 2^3 = 8. The average pool layer averages 2
    # consecutive of these outputs, thus generating an output for 16 consecutive frames, and because it's temporal
    # stride is 1 then it's every 8 frames. We could obtain 16-frame clips with a stride of 8, however we would
    # re-compute a lot of outputs for those 8 frames in common. I think it's better we make the clips the largest
    # possible and have batch size of 1. The only issue is that this makes each data loader worker load a lot of frames,
    # taking a lot of RAM, and also the start of the pipeline being quite slow if there are a lot of workers.
    #
    # This tradeoff should be considered then for setting the frames per clip, batch size, and the number of workers.
    # Also consider the video length (in our dataset they can be 30s or even 1m long).
    #
    # Note that taking 16-frame clips with a stride of 16 and then combining them (if they come from the same parent
    # video) before inputting them to the model would make that the next batch misses a feature (because it takes a
    # stride of 16 instead of 8). It could be solved by sampling more strategically, so that we recover those 8
    # frames back, and in the end need less RAM.
    #
    # Joao Carreira talks about it on GitHub (https://github.com/deepmind/kinetics-i3d/issues/97), and mentions it
    # on the paper https://arxiv.org/abs/1806.03863
    dataset = VideoDataset(args.input_path, clip_length_in_frames=args.frames_per_clip,
                           frames_between_clips=args.frames_per_clip - 8, frame_rate=25,
                           video_clips_num_workers=args.num_workers_video_clips)
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

        output = model(video_clip, return_logits=False).detach().cpu()

        for video_path, video_clip_output in zip(video_paths, output):
            outputs_by_path[video_path].append(video_clip_output)

        for video_path, is_last_clip_in_video_instance in zip(video_paths, is_last_clip_in_video):
            if is_last_clip_in_video_instance:  # Note the clips come in order for a given video.
                video_output = torch.cat(outputs_by_path.pop(video_path), dim=-1)

                filename = os.path.basename(video_path)
                filename_without_extension = filename.rsplit('.', maxsplit=1)[0]
                np.save(args.output_path / f"{filename_without_extension}.npy", video_output)


if __name__ == "__main__":
    main()
