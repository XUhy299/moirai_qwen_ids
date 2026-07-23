from __future__ import annotations

import unittest
from dataclasses import replace

from mqids.config import ExperimentConfig
from scripts.train import validate_run_name
from tests.probe_full_prompt_environment import recommend_batch_size


class ConfigAndEntrypointTests(unittest.TestCase):
    def test_no_qwen_dtt_is_rejected_during_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires the Qwen backbone"):
            replace(ExperimentConfig(projector="direct"), backbone="none", discrete_to_text=True)

    def test_invalid_run_names_cannot_escape_outputs(self) -> None:
        for name in ("../escape", "/absolute", r"..\escape", "has space"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_run_name(name)
        self.assertEqual(validate_run_name("cloud_dtt-a.seed2026"), "cloud_dtt-a.seed2026")

    def test_cuda_probe_does_not_recommend_a_batch_without_headroom(self) -> None:
        successes = [
            {"batch_size": 1, "peak_reserved_mb": 950.0},
            {"batch_size": 2, "peak_reserved_mb": 990.0},
        ]
        recommendation = recommend_batch_size(
            successes,
            device=__import__("torch").device("cuda"),
            total_memory_mb=1000.0,
            headroom_fraction=0.15,
        )
        self.assertIsNone(recommendation)


if __name__ == "__main__":
    unittest.main()
