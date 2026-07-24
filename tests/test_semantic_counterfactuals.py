from __future__ import annotations

import unittest

import torch
from torch import nn

from mqids.prompting import PromptAssembler
from mqids.semantics import VariableSemanticMap, describe_wadi_variable


class RecordingTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.encoded_texts: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        self.encoded_texts.append(text)
        return [1 + (ord(character) % 127) for character in text]


class SemanticCounterfactualTests(unittest.TestCase):
    names = (
        "1_AIT_002_PV",
        "2_FIT_001_PV",
        "2_MV_003_STATUS",
        "3_LT_001_PV",
    )

    def test_correct_semantics_keep_historical_payload_shape(self) -> None:
        semantic_map = VariableSemanticMap.from_names(self.names, style="compact")
        self.assertEqual(
            semantic_map.descriptions,
            tuple(describe_wadi_variable(name, style="compact") for name in self.names),
        )
        self.assertEqual(
            set(semantic_map.as_dict()),
            {"rule", "style", "variables"},
        )
        self.assertEqual(
            set(semantic_map.as_dict()["variables"][0]),
            {"name", "description"},
        )

    def test_id_only_removes_descriptions_but_preserves_names(self) -> None:
        semantic_map = VariableSemanticMap.from_names(
            self.names,
            style="compact",
            variant="id_only",
        )
        self.assertEqual(semantic_map.descriptions, ("",) * len(self.names))
        self.assertEqual(
            tuple(item["name"] for item in semantic_map.variables),
            self.names,
        )

    def test_shuffled_semantics_are_fixed_one_to_one_and_have_no_correct_assignments(self) -> None:
        first = VariableSemanticMap.from_names(
            self.names,
            style="compact",
            variant="shuffled",
            shuffle_seed=2026,
        )
        second = VariableSemanticMap.from_names(
            self.names,
            style="compact",
            variant="shuffled",
            shuffle_seed=2026,
        )
        different = VariableSemanticMap.from_names(
            self.names,
            style="compact",
            variant="shuffled",
            shuffle_seed=2027,
        )
        self.assertEqual(first.as_dict(), second.as_dict())
        self.assertEqual(first.sha256(), second.sha256())
        self.assertNotEqual(first.sha256(), different.sha256())
        source_names = tuple(
            str(item["description_source_name"])
            for item in first.variables
        )
        self.assertEqual(set(source_names), set(self.names))
        self.assertTrue(all(target != source for target, source in zip(self.names, source_names)))
        for item in first.variables:
            self.assertEqual(
                item["description"],
                describe_wadi_variable(str(item["description_source_name"]), style="compact"),
            )

    def test_id_only_prompt_omits_empty_parentheses(self) -> None:
        tokenizer = RecordingTokenizer()
        assembler = PromptAssembler(tokenizer, prefix="前缀", suffix="后缀")
        tokenizer.encoded_texts.clear()
        assembler.forward_variable_aligned(
            torch.zeros(1, 2, 4),
            nn.Embedding(256, 4),
            active_names=("A", "B"),
            active_descriptions=("", ""),
            semantic_style="compact",
            numeric_indices=(0, 1),
            discrete_indices=(),
            discrete_states=[()],
            window_length=64,
        )
        variable_texts = "".join(tokenizer.encoded_texts)
        self.assertIn("A，64步：", variable_texts)
        self.assertIn("B，64步：", variable_texts)
        self.assertNotIn("（）", variable_texts)


if __name__ == "__main__":
    unittest.main()
