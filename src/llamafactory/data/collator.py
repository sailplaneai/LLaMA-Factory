# Copyright 2024 OpenAccess AI Collective and the LlamaFactory team.
#
# This code is inspired by the OpenAccess AI Collective's axolotl library.
# https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/monkeypatch/utils.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Dict, Literal, Sequence

import torch
from transformers import DataCollatorForSeq2Seq
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from ..extras.constants import IGNORE_INDEX


def prepare_4d_attention_mask(attention_mask_with_indices: "torch.Tensor", dtype: "torch.dtype") -> "torch.Tensor":
    r"""
    Expands the attention mask with indices from (batch_size, seq_len) to (batch_size, 1, seq_len, seq_len),
    while handles packed sequences and transforms the mask to lower triangular form to prevent future peeking.

    e.g.
    ```python
    # input
    [[1, 1, 2, 2, 2, 0]]
    # output
    [
        [
            [
                [o, x, x, x, x, x],
                [o, o, x, x, x, x],
                [x, x, o, x, x, x],
                [x, x, o, o, x, x],
                [x, x, o, o, o, x],
                [x, x, x, x, x, x],
            ]
        ]
    ]
    ```
    where `o` equals to `0.0`, `x` equals to `min_dtype`.
    """
    bsz, seq_len = attention_mask_with_indices.size()
    min_dtype = torch.finfo(dtype).min
    expanded_mask = attention_mask_with_indices[:, None, None, :].expand(bsz, 1, seq_len, seq_len)
    # Create a binary mask from the original mask where zeros remain zeros and all other values are set to one
    padding_mask = torch.where(expanded_mask != 0, 1, 0)
    # Create a block-diagonal mask.
    attention_mask_4d = torch.eq(expanded_mask, expanded_mask.transpose(-1, -2)).int() * padding_mask
    # Use the lower triangular mask to zero out the upper triangular part
    attention_mask_4d *= torch.tril(torch.ones((seq_len, seq_len), dtype=torch.long))
    # Invert the attention mask.
    attention_mask_4d = torch.where(attention_mask_4d != 0, torch.tensor(0, dtype=dtype), min_dtype)
    return attention_mask_4d


@dataclass
class SFTDataCollatorWith4DAttentionMask(DataCollatorForSeq2Seq):
    r"""
    Data collator for 4d attention mask.
    """

    block_diag_attn: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] = "eager"
    compute_dtype: "torch.dtype" = torch.float32

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        features = super().__call__(features)
        if self.block_diag_attn and self.attn_implementation != "flash_attention_2":
            features["attention_mask"] = prepare_4d_attention_mask(features["attention_mask"], self.compute_dtype)

        return features


@dataclass
class PairwiseDataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        for key in ("chosen", "rejected"):
            for feature in features:
                target_feature = {
                    "input_ids": feature["{}_input_ids".format(key)],
                    "attention_mask": feature["{}_attention_mask".format(key)],
                    "labels": feature["{}_labels".format(key)],
                }
                if "pixel_values" in feature:
                    target_feature["pixel_values"] = feature["pixel_values"]

                if "{}_token_type_ids".format(key) in feature:
                    target_feature["token_type_ids"] = feature["{}_token_type_ids".format(key)]

                concatenated_features.append(target_feature)

        return super().__call__(concatenated_features)


@dataclass
class KTODataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for KTO data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, "torch.Tensor"]:
        target_features = []
        kl_features = []
        kto_tags = []
        for feature in features:
            target_feature = {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "labels": feature["labels"],
            }
            kl_feature = {
                "input_ids": feature["kl_input_ids"],
                "attention_mask": feature["kl_attention_mask"],
                "labels": feature["kl_labels"],
            }
            if "pixel_values" in feature:
                target_feature["pixel_values"] = feature["pixel_values"]

            if "token_type_ids" in feature:
                target_feature["token_type_ids"] = feature["token_type_ids"]
                kl_feature["token_type_ids"] = feature["kl_token_type_ids"]

            target_features.append(target_feature)
            kl_features.append(kl_feature)
            kto_tags.append(feature["kto_tags"])

        batch = super().__call__(target_features)
        kl_batch = super().__call__(kl_features)
        batch["kl_input_ids"] = kl_batch["input_ids"]
        batch["kl_attention_mask"] = kl_batch["attention_mask"]
        batch["kl_labels"] = kl_batch["labels"]
        if "token_type_ids" in batch:
            batch["kl_token_type_ids"] = kl_batch["token_type_ids"]

        batch["kto_tags"] = torch.tensor(kto_tags)
        return batch

@dataclass
class VMDataCollatorForSeq2Seq(DataCollatorForSeq2Seq):

    # tokenizer: PreTrainedTokenizerBase
    # model: Optional[Any] = None
    # padding: Union[bool, str, PaddingStrategy] = True
    # max_length: Optional[int] = None
    # pad_to_multiple_of: Optional[int] = None
    # label_pad_token_id: int = -100
    # return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

            # Padding Q to the longest sequence in the batch
            if "Q" in features[0]:   # float32
                # max_length_Q = max(len(feature["Q"]) for feature in features)
                for feature in features:
                    remainder = [IGNORE_INDEX] * (max_label_length - len(feature["Q"]))  # Assuming IGNORE_INDEX as padding value for Q
                    
                    if isinstance(feature["Q"], list):
                        feature["Q"] = (
                            feature["Q"] + remainder if padding_side == "right" else remainder + feature["Q"]
                        )
                    elif padding_side == "right":
                        feature["Q"] = np.concatenate([feature["Q"], remainder]).astype(np.float32)
                    else:
                        feature["Q"] = np.concatenate([remainder, feature["Q"]]).astype(np.float32)

        features = pad_without_fast_tokenizer_warning(  # only padding input_ids and attention_mask
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features
