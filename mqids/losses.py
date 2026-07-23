"""Primary classifier loss plus restricted two-token Qwen verbalizer loss."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DualObjectiveLoss(nn.Module):
    def __init__(self, vocab_weight: float = 0.1, classifier_weight: float = 1.0) -> None:
        super().__init__()
        if vocab_weight < 0 or classifier_weight < 0:
            raise ValueError("Loss weights cannot be negative")
        if vocab_weight == 0 and classifier_weight == 0:
            raise ValueError("At least one loss weight must be positive")
        self.vocab_weight = float(vocab_weight)
        self.classifier_weight = float(classifier_weight)

    def forward(
        self,
        classifier_logits: torch.Tensor,
        verbalizer_logits: torch.Tensor | None,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        classifier_loss = F.cross_entropy(classifier_logits.float(), labels)
        if verbalizer_logits is None:
            if self.vocab_weight != 0:
                raise ValueError("A positive verbalizer weight requires verbalizer logits")
            verbalizer_loss = classifier_loss.new_zeros(())
        else:
            verbalizer_loss = F.cross_entropy(verbalizer_logits.float(), labels)
        total = self.classifier_weight * classifier_loss + self.vocab_weight * verbalizer_loss
        return {
            "loss": total,
            "classifier_loss": classifier_loss.detach(),
            "verbalizer_loss": verbalizer_loss.detach(),
        }
