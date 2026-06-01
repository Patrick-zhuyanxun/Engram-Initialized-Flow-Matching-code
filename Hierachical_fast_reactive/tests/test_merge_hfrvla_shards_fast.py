from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.merge_hfrvla_shards_fast import _link_data_files, _merge_episodes


def _write_episode_file(root: Path, filename: str, rows: list[dict]) -> None:
    out = root / "meta/episodes/chunk-000" / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)


def test_merge_episodes_reads_all_episode_metadata_files(tmp_path):
    shard_a = tmp_path / "shard_a"
    shard_b = tmp_path / "shard_b"
    base_row = {
        "length": 10,
        "dataset_from_index": 0,
        "dataset_to_index": 10,
        "data/file_index": 0,
    }
    _write_episode_file(shard_a, "file-000.parquet", [{**base_row, "episode_index": 0}])
    _write_episode_file(
        shard_a,
        "file-001.parquet",
        [{**base_row, "episode_index": 1, "dataset_from_index": 10, "dataset_to_index": 20}],
    )
    _write_episode_file(shard_b, "file-000.parquet", [{**base_row, "episode_index": 0}])

    merged = _merge_episodes(
        [shard_a, shard_b],
        ep_off=[0, 2, 3],
        fr_off=[0, 20, 30],
        fi_off=[0, 2, 3],
        out_eps_dir=tmp_path / "out/meta/episodes/chunk-000",
    )

    assert merged["episode_index"].tolist() == [0, 1, 2]
    assert merged["dataset_from_index"].tolist() == [0, 10, 20]
    assert merged["dataset_to_index"].tolist() == [10, 20, 30]


def test_link_data_files_rewrites_global_episode_and_frame_indices(tmp_path):
    shard_a = tmp_path / "shard_a"
    shard_b = tmp_path / "shard_b"
    for root in (shard_a, shard_b):
        data_dir = root / "data/chunk-000"
        data_dir.mkdir(parents=True)
        table = pa.table(
            {
                "episode_index": pa.array([0, 0], type=pa.int64()),
                "index": pa.array([0, 1], type=pa.int64()),
                "task_index": pa.array([0, 1], type=pa.int64()),
            }
        )
        pq.write_table(table, data_dir / "file-000.parquet")

    out_dir = tmp_path / "out/data/chunk-000"
    _link_data_files(
        [shard_a, shard_b],
        ep_offsets=[0, 3, 4],
        fr_offsets=[0, 20, 22],
        fi_offsets=[0, 1, 2],
        remaps=[{0: 0, 1: 1}, {0: 2, 1: 3}],
        out_data_dir=out_dir,
        use_copy=False,
    )

    out_a = pq.read_table(out_dir / "file-000.parquet").to_pydict()
    out_b = pq.read_table(out_dir / "file-001.parquet").to_pydict()

    assert out_a["episode_index"] == [0, 0]
    assert out_a["index"] == [0, 1]
    assert out_a["task_index"] == [0, 1]
    assert out_b["episode_index"] == [3, 3]
    assert out_b["index"] == [20, 21]
    assert out_b["task_index"] == [2, 3]
