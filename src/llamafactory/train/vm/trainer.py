# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
from transformers import Trainer
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.nn as nn

from ...extras.logging import get_logger
from ..callbacks import FixValueHeadModelCallback, PissaConvertCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimzer, create_custom_scheduler
from ...extras.constants import IGNORE_INDEX


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class ValueTrainer(Trainer):
    r"""
    Inherits Trainer to compute pairwise loss.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.can_return_loss = True  # override property to return eval_loss
        self.add_callback(FixValueHeadModelCallback)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.pissa_convert:
            self.add_callback(PissaConvertCallback)

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def get_model_attribute(self, model, attribute_name):
        if isinstance(model, nn.parallel.DistributedDataParallel):
            model = model.module
        return getattr(model, attribute_name)

    def compute_loss(
            self, model: "PreTrainedModel", inputs: Dict[str, torch.Tensor], return_outputs: bool = False
        ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:

            # Compute rewards
            Q = inputs.get("Q", None)
            if Q is not None:
                del inputs["Q"]

            labels = inputs.get("labels", None)
            if labels is not None:
                del inputs["labels"]

            mask = Q.ne(IGNORE_INDEX)

            lm_logits, loss, values = model(**inputs, output_hidden_states=True, return_dict=True)
            values = torch.tanh(values)

            if loss is None:
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                if torch.all(shift_labels==IGNORE_INDEX):
                    loss_fct = CrossEntropyLoss(reduction='sum')
                else:
                    loss_fct = CrossEntropyLoss()
                if isinstance(model, nn.parallel.DistributedDataParallel):
                    shift_logits = shift_logits.view(-1, model.module.pretrained_model.config.vocab_size)
                else:
                    shift_logits = shift_logits.view(-1, model.pretrained_model.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)

            assert not torch.isnan(loss) and Q is not None

            Q = Q.type_as(values)
            masked_values = torch.where(mask, values, Q)
            value_loss = F.mse_loss(masked_values, Q, reduction='sum') / (mask.sum() + 1e-3)
            all_losses = loss + 0.1 * value_loss

            if return_outputs:
                return all_losses, [all_losses, loss, value_loss, masked_values, Q]
            return all_losses#, value_loss

    def save_predictions(self, predict_results: "PredictionOutput") -> None:
        r"""
        Saves model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")
        chosen_scores, rejected_scores = predict_results.predictions

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for c_score, r_score in zip(chosen_scores, rejected_scores):
                res.append(json.dumps({"chosen": round(float(c_score), 2), "rejected": round(float(r_score), 2)}))

            writer.write("\n".join(res))
