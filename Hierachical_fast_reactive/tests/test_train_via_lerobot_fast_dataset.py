from types import SimpleNamespace

import pytest

from scripts.train_via_lerobot import (
    HFRVLA_OFFLINE_REQUIRED_COLUMNS,
    _apply_lr_scale,
    _dataset_backend,
    _is_offline_hfrvla,
    _load_offline_hfrvla_hf_dataset,
    _offline_hfrvla_delta_timestamps,
    _offline_hfrvla_features,
    _patched_make_dataset,
    _select_offline_hfrvla_columns,
)


def test_offline_hfrvla_detection():
    cfg = SimpleNamespace(policy=SimpleNamespace(type="hfrvla", offline_training_mode=True))

    assert _is_offline_hfrvla(cfg)


def test_dataset_backend_defaults_to_lerobot(monkeypatch):
    monkeypatch.delenv("HFRVLA_DATASET_BACKEND", raising=False)

    assert _dataset_backend() == "lerobot"


def test_dataset_backend_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("HFRVLA_DATASET_BACKEND", "bad")

    with pytest.raises(ValueError, match="HFRVLA_DATASET_BACKEND"):
        _dataset_backend()


def test_fastcache_make_dataset_requires_root(monkeypatch):
    monkeypatch.setenv("HFRVLA_DATASET_BACKEND", "fastcache")
    monkeypatch.delenv("HFRVLA_FASTCACHE_ROOT", raising=False)
    cfg = SimpleNamespace(
        policy=SimpleNamespace(type="hfrvla", offline_training_mode=True, seq_len=4)
    )

    with pytest.raises(ValueError, match="HFRVLA_FASTCACHE_ROOT"):
        _patched_make_dataset(cfg)


def test_offline_delta_timestamps_excludes_images():
    cfg = SimpleNamespace(
        policy=SimpleNamespace(
            type="hfrvla",
            offline_training_mode=True,
            observation_delta_indices=[-1, 0],
            action_delta_indices=[-1, 0],
            reward_delta_indices=None,
        )
    )
    meta = SimpleNamespace(
        fps=10,
        features={
            "observation.images.image": {},
            "observation.images.image2": {},
            "observation.state": {},
            "observation.extra.dino_patches": {},
            "action": {},
            "index": {},
        },
    )

    delta = _offline_hfrvla_delta_timestamps(cfg, meta)

    assert "observation.images.image" not in delta
    assert "observation.images.image2" not in delta
    assert delta["observation.state"] == [-0.1, 0.0]
    assert delta["observation.extra.dino_patches"] == [-0.1, 0.0]
    assert delta["action"] == [-0.1, 0.0]


def test_offline_hfrvla_features_excludes_images_before_parquet_load():
    features = {
        "observation.images.image": {"dtype": "image"},
        "observation.images.image2": {"dtype": "image"},
        "observation.state": {"dtype": "float32"},
        "observation.extra.a_base": {"dtype": "float32"},
        "observation.extra.k_idx_norm": {"dtype": "float32"},
        "observation.extra.z_goal": {"dtype": "float32"},
        "observation.extra.z_phase": {"dtype": "float32"},
        "observation.extra.dino_patches": {"dtype": "float32"},
        "observation.extra.contact_label": {"dtype": "float32"},
        "action": {"dtype": "float32"},
        "index": {"dtype": "int64"},
        "frame_index": {"dtype": "int64"},
        "episode_index": {"dtype": "int64"},
        "task_index": {"dtype": "int64"},
        "timestamp": {"dtype": "float32"},
    }

    selected = _offline_hfrvla_features(features)

    assert "observation.images.image" not in selected
    assert "observation.images.image2" not in selected
    assert HFRVLA_OFFLINE_REQUIRED_COLUMNS.issubset(set(selected))


def test_offline_loader_projects_columns_before_casting_parquet(monkeypatch, tmp_path):
    parquet_dir = tmp_path / "data" / "chunk-000"
    parquet_dir.mkdir(parents=True)
    (parquet_dir / "file-000.parquet").write_bytes(b"not read by mocked from_parquet")
    captured = {}

    class _HF:
        def set_transform(self, transform):
            captured["transform"] = transform

    def _from_parquet(paths, *, features, columns, filters):
        captured["paths"] = paths
        captured["features"] = features
        captured["columns"] = columns
        captured["filters"] = filters
        return _HF()

    monkeypatch.setattr(
        "scripts.train_via_lerobot.dataset_reader_module.datasets.Dataset.from_parquet",
        _from_parquet,
    )
    features = {
        "observation.images.image": {"dtype": "image", "shape": (256, 256, 3)},
        "observation.images.image2": {"dtype": "image", "shape": (256, 256, 3)},
        "observation.state": {"dtype": "float32", "shape": (8,)},
        "observation.extra.a_base": {"dtype": "float32", "shape": (7,)},
        "observation.extra.k_idx_norm": {"dtype": "float32", "shape": (1,)},
        "observation.extra.z_goal": {"dtype": "float32", "shape": (960,)},
        "observation.extra.z_phase": {"dtype": "float32", "shape": (480,)},
        "observation.extra.dino_patches": {"dtype": "float32", "shape": (196, 384)},
        "observation.extra.contact_label": {"dtype": "float32", "shape": (1,)},
        "action": {"dtype": "float32", "shape": (7,)},
        "index": {"dtype": "int64", "shape": (1,)},
        "frame_index": {"dtype": "int64", "shape": (1,)},
        "episode_index": {"dtype": "int64", "shape": (1,)},
        "task_index": {"dtype": "int64", "shape": (1,)},
        "timestamp": {"dtype": "float32", "shape": (1,)},
    }
    reader = SimpleNamespace(
        root=tmp_path,
        episodes=None,
        _meta=SimpleNamespace(features=features),
    )

    _load_offline_hfrvla_hf_dataset(reader)

    assert "observation.images.image" not in captured["columns"]
    assert "observation.images.image2" not in captured["columns"]
    assert HFRVLA_OFFLINE_REQUIRED_COLUMNS.issubset(set(captured["columns"]))


def test_select_offline_columns_removes_images_from_hf_dataset():
    calls = []

    class _HF:
        column_names = [
            "observation.images.image",
            "observation.images.image2",
            "observation.state",
            "observation.extra.a_base",
            "observation.extra.k_idx_norm",
            "observation.extra.z_goal",
            "observation.extra.z_phase",
            "observation.extra.dino_patches",
            "observation.extra.contact_label",
            "action",
            "index",
            "frame_index",
            "episode_index",
            "task_index",
            "timestamp",
        ]

        def select_columns(self, columns):
            calls.append(list(columns))
            return {"selected": list(columns)}

    dataset = SimpleNamespace(reader=SimpleNamespace(hf_dataset=_HF()))

    _select_offline_hfrvla_columns(dataset)

    selected = calls[0]
    assert "observation.images.image" not in selected
    assert "observation.images.image2" not in selected
    assert HFRVLA_OFFLINE_REQUIRED_COLUMNS.issubset(set(selected))
    assert dataset.reader.hf_dataset == {"selected": selected}


def test_apply_lr_scale_reaches_accelerate_wrapped_scheduler():
    optimizer = SimpleNamespace(param_groups=[{"lr": 3e-4, "initial_lr": 3e-4}])
    inner_scheduler = SimpleNamespace(base_lrs=[3e-4], _last_lr=[3e-4])
    wrapped_scheduler = SimpleNamespace(scheduler=inner_scheduler)

    changed = _apply_lr_scale(optimizer, wrapped_scheduler, factor=0.1)

    assert changed is True
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)
    assert optimizer.param_groups[0]["initial_lr"] == pytest.approx(3e-5)
    assert inner_scheduler.base_lrs == pytest.approx([3e-5])
    assert inner_scheduler._last_lr == pytest.approx([3e-5])
