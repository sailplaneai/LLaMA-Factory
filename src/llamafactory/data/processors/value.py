# Copyright 2024 the LlamaFactory team.
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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from .processor_utils import get_paligemma_token_type_ids, get_pixel_values, infer_seqlen

import json
if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..template import Template


logger = get_logger(__name__)


def preprocess_value_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X ` and labels with format `X <eos> `
    model_inputs = {"input_ids": [], "attention_mask": [], "Q": [], "labels": []}

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:  # prompt only one, response only one
            continue

        message = examples["prompt"][i] + [{"role": 'assistant', 'content': ""}]
        input_ids, _ = template.encode_oneturn(tokenizer, message, examples["system"][i], examples["tools"][i])

        if data_args.train_on_prompt:
            print("train_on_prompt")
            source_mask = input_ids
            Q = [IGNORE_INDEX] * len(input_ids)

        else:
            source_mask = [IGNORE_INDEX] * len(input_ids)
            Q = [IGNORE_INDEX] * len(input_ids)

        # input_ids += source_ids + target_ids
        # labels += source_mask + target_ids
        labels = source_mask
        
        multistep_response = json.loads(examples["response"][i][0]['content'])
        response_state = multistep_response[-1]['Q']  # last Q
        for sub_response in multistep_response:
            if len(sub_response['step']) == 0:
                print(sub_response['step'])
            sub_message = [{"role": 'user', 'content': ""}] + [{"role": 'assistant', 'content': sub_response['step'].strip() + "\n"}]
            sub_Q = float(sub_response['Q'])
            _, sub_response_ids = template.encode_oneturn(tokenizer, sub_message, examples["system"][i], examples["tools"][i])
            
            # sub_response_ids = sub_response_ids[:-1]  # discard the 1000001
            # to make sure the sentence ends with \n instead of <eos>
            # our value model predicts the v based on '\n'

            input_ids += sub_response_ids
            Q += [IGNORE_INDEX] * (len(sub_response_ids) - 1) + [sub_Q]
            labels += sub_response_ids

            if len(input_ids) > data_args.cutoff_len:
                break

        if template.efficient_eos:  # vanilla template will go into
            input_ids += [tokenizer.eos_token_id]
            Q += [IGNORE_INDEX]
            labels += [tokenizer.eos_token_id]

        if len(input_ids) > data_args.cutoff_len:
            input_ids = input_ids[:data_args.cutoff_len]
            Q = Q[:data_args.cutoff_len]
            labels = labels[:data_args.cutoff_len]


        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["Q"].append(Q)


        if response_state == -1:
            model_inputs["labels"].append([IGNORE_INDEX] * len(labels))
        elif response_state == 1:
            model_inputs["labels"].append(labels)
        else:
            assert False, response_state

    return model_inputs
