---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
configs:
- config_name: default
  data_files: data/*/*.parquet
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Description



- **Homepage:** [More Information Needed]
- **Paper:** [More Information Needed]
- **License:** apache-2.0

## Dataset Structure

[meta/info.json](meta/info.json):
```json
{
    "codebase_version": "v3.0",
    "robot_type": "panda",
    "total_episodes": 1693,
    "total_frames": 273465,
    "total_tasks": 40,
    "chunks_size": 1000,
    "fps": 10.0,
    "splits": {
        "train": "0:1693"
    },
    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    "features": {
        "observation.images.image": {
            "dtype": "image",
            "shape": [
                256,
                256,
                3
            ],
            "names": [
                "height",
                "width",
                "channel"
            ],
            "fps": 10.0
        },
        "observation.images.image2": {
            "dtype": "image",
            "shape": [
                256,
                256,
                3
            ],
            "names": [
                "height",
                "width",
                "channel"
            ],
            "fps": 10.0
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [
                8
            ],
            "names": [
                "state"
            ],
            "fps": 10.0
        },
        "action": {
            "dtype": "float32",
            "shape": [
                7
            ],
            "names": [
                "actions"
            ],
            "fps": 10.0
        },
        "timestamp": {
            "dtype": "float32",
            "shape": [
                1
            ],
            "names": null,
            "fps": 10.0
        },
        "frame_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null,
            "fps": 10.0
        },
        "episode_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null,
            "fps": 10.0
        },
        "index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null,
            "fps": 10.0
        },
        "task_index": {
            "dtype": "int64",
            "shape": [
                1
            ],
            "names": null,
            "fps": 10.0
        }
    },
    "data_files_size_in_mb": 100,
    "video_files_size_in_mb": 500
}
```


## Citation

**BibTeX:**

```bibtex
[More Information Needed]
```