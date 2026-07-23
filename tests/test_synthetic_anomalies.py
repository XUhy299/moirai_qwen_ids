from __future__ import annotations

import unittest
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from mqids.synthetic_anomalies import (
    OPERATOR_NAMES,
    FullSyntheticAnomalyGenerator,
    SyntheticAnomalyConfig,
    fit_synthetic_statistics,
)
from mqids.data import SyntheticWindowDataset, build_stratified_epoch_schedules
from mqids.data import infer_channel_metadata, infer_discrete_state_vocabulary
from mqids.switch_templates import build_discrete_text


class SyntheticAnomalyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(7)
        steps = 600
        time = np.arange(steps, dtype=np.float64)
        cls.source = np.column_stack(
            (
                np.sin(time / 13.0) + 0.03 * rng.normal(size=steps),
                np.cos(time / 19.0) + 0.03 * rng.normal(size=steps),
                ((time // 17) % 2).astype(np.float64),
                ((time // 29) % 3).astype(np.float64),
            )
        )
        cls.stats = fit_synthetic_statistics(
            cls.source,
            active_indices=(0, 1, 2, 3),
            type_ids=(0, 0, 1, 2),
            sample_size=600,
        )
        cls.generator = FullSyntheticAnomalyGenerator(
            cls.source,
            cls.stats,
            window_length=64,
            config=SyntheticAnomalyConfig(),
        )

    def test_every_operator_changes_endpoint_and_preserves_shape(self) -> None:
        for index, operator in enumerate(OPERATOR_NAMES):
            with self.subTest(operator=operator):
                sample = self.generator.generate(
                    source_endpoint=300 + index * 10,
                    rng=np.random.default_rng(100 + index),
                    operator=operator,
                )
                self.assertEqual(sample.window.shape, (64, 4))
                self.assertEqual(sample.point_mask.shape, (64,))
                self.assertEqual(sample.channel_mask.shape, (4,))
                self.assertTrue(sample.point_mask[-1])
                self.assertTrue(sample.channel_mask.any())
                self.assertTrue(np.isfinite(sample.window).all())

    def test_discrete_outputs_remain_in_training_vocabularies(self) -> None:
        rng = np.random.default_rng(1234)
        for endpoint in range(100, 500, 7):
            sample = self.generator.generate(endpoint, rng)
            for position in (2, 3):
                legal = np.asarray(self.stats.discrete_values[position])
                self.assertTrue(np.isin(sample.window[:, position], legal).all())

    def test_same_seed_is_reproducible(self) -> None:
        first = self.generator.generate(350, np.random.default_rng(99))
        second = self.generator.generate(350, np.random.default_rng(99))
        np.testing.assert_array_equal(first.window, second.window)
        np.testing.assert_array_equal(first.point_mask, second.point_mask)
        self.assertEqual(first.metadata, second.metadata)

    def test_epoch_stratified_schedules_are_reproducible_and_non_overlapping(self) -> None:
        operator_codes = np.repeat(
            np.arange(5, dtype=np.int64),
            np.asarray([300, 500, 300, 600, 300]),
        )
        first = build_stratified_epoch_schedules(
            operator_codes,
            samples_per_epoch=170,
            epochs=5,
            seed=2026,
        )
        second = build_stratified_epoch_schedules(
            operator_codes,
            samples_per_epoch=170,
            epochs=5,
            seed=2026,
        )
        self.assertEqual(len(first), 5)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
            self.assertEqual(len(left), 170)
            counts = np.bincount(operator_codes[left], minlength=5)
            np.testing.assert_array_equal(counts, np.asarray([26, 43, 25, 51, 25]))
        combined = np.concatenate(first)
        self.assertEqual(len(np.unique(combined)), 850)

    def test_unknown_discrete_state_reports_inverse_scaled_raw_value(self) -> None:
        train = np.asarray([[0.0], [1.0]], dtype=np.float64)
        metadata = infer_channel_metadata(train, ("1_LS_001_STATUS",))
        vocabulary = infer_discrete_state_vocabulary(
            train,
            metadata,
            source="unit-test",
            scaler_mean=np.asarray([10.0]),
            scaler_scale=np.asarray([2.0]),
        )
        text = build_discrete_text(torch.tensor([[0.0, 3.0]]), vocabulary)
        self.assertEqual(text, ("未知状态（原始值=16）",))

    def test_offline_package_matches_training_dataset_interface(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            samples = [
                self.generator.generate(300 + index, np.random.default_rng(500 + index))
                for index in range(5)
            ]
            windows = np.stack([sample.window for sample in samples])
            point_masks = np.stack([sample.point_mask for sample in samples])
            channel_masks = np.stack([sample.channel_mask for sample in samples])
            np.save(package / "synthetic_windows.npy", windows)
            np.save(package / "point_masks.npy", point_masks)
            np.save(package / "channel_masks.npy", channel_masks)
            np.save(package / "endpoint_labels.npy", np.ones(5, dtype=np.uint8))
            np.save(package / "source_endpoints.npy", np.arange(300, 305, dtype=np.int64))
            np.save(package / "operator_codes.npy", np.arange(5, dtype=np.uint8))
            train_hash = hashlib.sha256(b"train").hexdigest()
            (package / "dataset_summary.json").write_text(
                json.dumps(
                    {
                        "method": "full_five_type_mixture",
                        "validation_or_test_arrays_opened": False,
                        "input_x_sha256": train_hash,
                    }
                ),
                encoding="utf-8",
            )
            (package / "channel_metadata.json").write_text(
                json.dumps(
                    {
                        "active_indices": [0, 1, 2, 3],
                        "type_ids": [0, 0, 1, 2],
                    }
                ),
                encoding="utf-8",
            )
            dataset = SyntheticWindowDataset(
                package,
                expected_window_length=64,
                expected_active_indices=(0, 1, 2, 3),
                expected_type_ids=(0, 0, 1, 2),
                expected_train_x_sha256=train_hash,
                expected_train_length=600,
            )
            item = dataset[0]
            self.assertEqual(tuple(item["window"].shape), (64, 4))
            self.assertEqual(int(item["label"]), 1)
            self.assertEqual(int(item["endpoint"]), 300)
            dataset.close()


if __name__ == "__main__":
    unittest.main()
