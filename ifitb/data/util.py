import glob
import re
from datetime import timedelta

import numpy as np
import torch

from ifitb.util.file_utils import cached_path

# We ignore the sub-second component as the filenames are truncated.
RE_TIME = re.compile(r"^(?P<hours>\d+?):(?P<minutes>\d+?):(?P<seconds>\d+?)(?:\.\d+)?$")


# Use `timedelta` as it represents arbitrary time. `time` represents an arbitrary hour of *a day* (limited to 24h).
# `timedelta` in this context can be thought as the time between the video beginning and a certain timestamp.

# From https://stackoverflow.com/a/4628148/1165181
def _parse_time(s: str) -> timedelta:
    match = RE_TIME.match(s)
    assert match, f"The timestamp {s} could not be parsed"
    return timedelta(**{k: int(v) for k, v in match.groupdict().items()})


def _get_video_feature_filename(video_id: str, start_time: timedelta, end_time: timedelta) -> str:
    return f"{video_id}+{start_time}+{end_time}.npy"


def _get_video_features(visual_data_dir: str, filename: str) -> torch.Tensor:
    if visual_data_dir.endswith("/"):
        visual_data_dir = visual_data_dir[:-1]

    try:
        path = next(glob.iglob(f"{visual_data_dir}/*/{filename}"))
    except StopIteration as e:
        raise ValueError(f"Video features file not found: {filename}") from e

    return torch.from_numpy(np.load(path)).T


def get_video_features(visual_data_dir: str, video_id: str, start_time_str: str, end_time_str: str) -> torch.Tensor:
    visual_data_dir = cached_path(visual_data_dir)

    filename = _get_video_feature_filename(video_id, start_time=_parse_time(start_time_str),
                                           end_time=_parse_time(end_time_str))
    return _get_video_features(visual_data_dir, filename)
