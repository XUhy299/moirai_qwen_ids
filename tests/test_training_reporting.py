from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mqids.training import write_run_markdown


class TrainingReportingTests(unittest.TestCase):
    def test_verbalizer_only_report_uses_vocab_metrics_everywhere(self) -> None:
        history = [
            {
                "epoch": 1,
                "train": {"loss": 0.5, "seconds": 1.0},
                "dev": {
                    "roc_auc": 0.99,
                    "pr_auc": 0.98,
                    "precision": 0.97,
                    "recall": 0.96,
                    "f1": 0.95,
                    "false_positive_rate": 0.94,
                    "false_negative_rate": 0.93,
                    "threshold": 0.92,
                    "vocab_roc_auc": 0.71,
                    "vocab_pr_auc": 0.21,
                    "vocab_precision": 0.31,
                    "vocab_recall": 0.41,
                    "vocab_f1": 0.35,
                    "vocab_false_positive_rate": 0.11,
                    "vocab_false_negative_rate": 0.59,
                    "vocab_threshold": 0.61,
                    "seconds": 2.0,
                },
                "selection_score": 0.21,
                "peak_cuda_memory_mb": 100.0,
            },
            {
                "epoch": 2,
                "train": {"loss": 0.4, "seconds": 1.1},
                "dev": {
                    "roc_auc": 0.89,
                    "pr_auc": 0.88,
                    "precision": 0.87,
                    "recall": 0.86,
                    "f1": 0.85,
                    "false_positive_rate": 0.84,
                    "false_negative_rate": 0.83,
                    "threshold": 0.82,
                    "vocab_roc_auc": 0.72,
                    "vocab_pr_auc": 0.32,
                    "vocab_precision": 0.42,
                    "vocab_recall": 0.52,
                    "vocab_f1": 0.47,
                    "vocab_false_positive_rate": 0.12,
                    "vocab_false_negative_rate": 0.48,
                    "vocab_threshold": 0.62,
                    "seconds": 2.1,
                },
                "selection_score": 0.32,
                "peak_cuda_memory_mb": 101.0,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            write_run_markdown(
                output_dir,
                config={
                    "classifier_loss_weight": 0.0,
                    "vocab_loss_weight": 1.0,
                    "epochs": 2,
                },
                protocol={},
                environment={},
                history=history,
                status="completed",
            )
            report = (output_dir / "results.md").read_text(encoding="utf-8")

        self.assertIn("| 主评估输出 | verbalizer |", report)
        self.assertIn(
            "| 2 | 0.4000 | 0.7200 | 0.3200 | 0.4200 | 0.5200 | "
            "0.4700 | 0.1200 | 0.6200 | 1.1000 | 2.1000 |",
            report,
        )
        self.assertIn("- Epoch：2", report)
        self.assertIn("- 主评估输出：verbalizer", report)
        self.assertIn("- PR-AUC：0.3200", report)
        self.assertIn("- Precision / Recall / F1：0.4200 / 0.5200 / 0.4700", report)
        self.assertIn("- FPR / FNR：0.1200 / 0.4800", report)
        self.assertNotIn("- PR-AUC：0.8800", report)

    def test_classifier_report_keeps_classifier_metrics(self) -> None:
        history = [
            {
                "epoch": 1,
                "train": {"loss": 0.5, "seconds": 1.0},
                "dev": {
                    "roc_auc": 0.91,
                    "pr_auc": 0.81,
                    "precision": 0.71,
                    "recall": 0.61,
                    "f1": 0.66,
                    "false_positive_rate": 0.09,
                    "false_negative_rate": 0.39,
                    "threshold": 0.51,
                    "vocab_roc_auc": 0.2,
                    "vocab_pr_auc": 0.1,
                    "vocab_precision": 0.1,
                    "vocab_recall": 0.1,
                    "vocab_f1": 0.1,
                    "vocab_false_positive_rate": 0.8,
                    "vocab_false_negative_rate": 0.9,
                    "vocab_threshold": 0.2,
                    "seconds": 2.0,
                },
                "selection_score": 0.81,
                "peak_cuda_memory_mb": 100.0,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            write_run_markdown(
                output_dir,
                config={
                    "classifier_loss_weight": 1.0,
                    "vocab_loss_weight": 0.0,
                    "epochs": 1,
                },
                protocol={},
                environment={},
                history=history,
                status="completed",
            )
            report = (output_dir / "results.md").read_text(encoding="utf-8")

        self.assertIn("| 主评估输出 | classifier |", report)
        self.assertIn("- 主评估输出：classifier", report)
        self.assertIn("- PR-AUC：0.8100", report)
        self.assertIn("- Precision / Recall / F1：0.7100 / 0.6100 / 0.6600", report)
        self.assertIn("- FPR / FNR：0.0900 / 0.3900", report)


if __name__ == "__main__":
    unittest.main()
