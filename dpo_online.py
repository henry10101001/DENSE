# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
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

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, GenerationConfig

from trl import (
    LogCompletionsCallback,
    OnlineDPOTrainer,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from judge import DictionaryJudge, OpenAIJudge  # my custom judge implementations

# global configuration variables
MODEL = 'google/gemma-3-4b-it'
REWARD_MODEL_PATH = None  # set to path if using reward model
DATASET_NAME = 'trl-lib/tldr'
DATASET_CONFIG = None
DATASET_TRAIN_SPLIT = 'train'
DATASET_TEST_SPLIT = 'test'
OUTPUT_DIR = 'gemma-3-4b-dpo'
LEARNING_RATE = 5.0e-7
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 16
WARMUP_RATIO = 0.1
MISSING_EOS_PENALTY = 1.0
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
EVAL_STRATEGY = 'steps'
EVAL_STEPS = 500
USE_PEFT = False
JUDGE = None  # set to 'dictionary' or 'openai' if using judge
PUSH_TO_HUB = False
GRADIENT_CHECKPOINTING = True

JUDGES = {"dictionary": DictionaryJudge, "openai": OpenAIJudge}

if __name__ == "__main__":
    # create a simple config object to hold training arguments
    class TrainingConfig:
        def __init__(self):
            self.reward_model_path = REWARD_MODEL_PATH
            self.judge = JUDGE
            self.gradient_checkpointing = GRADIENT_CHECKPOINTING
            self.gradient_checkpointing_kwargs = {"use_reentrant": True}
            self.eval_strategy = EVAL_STRATEGY
            self.eval_steps = EVAL_STEPS
            self.max_new_tokens = MAX_NEW_TOKENS
            self.temperature = TEMPERATURE
            self.output_dir = OUTPUT_DIR
            self.push_to_hub = PUSH_TO_HUB
            self.learning_rate = LEARNING_RATE
            self.per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE
            self.gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS
            self.warmup_ratio = WARMUP_RATIO
            self.missing_eos_penalty = MISSING_EOS_PENALTY

    class ModelConfig:
        def __init__(self):
            self.model_name_or_path = MODEL
            self.model_revision = None
            self.attn_implementation = None
            self.torch_dtype = "auto"
            self.trust_remote_code = True

    training_args = TrainingConfig()
    model_args = ModelConfig()

    # set up model dtype and quantization
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args) if USE_PEFT else None  # config model quantization (if specified)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,  # disable cache with gradient checkpointing
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    # load main language model for training
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )

    # load reward model (if specified)
    if training_args.reward_model_path is not None:
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            training_args.reward_model_path,
            num_labels=1,  # single score output
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
        reward_tokenizer = AutoTokenizer.from_pretrained(
            training_args.reward_model_path,
            trust_remote_code=model_args.trust_remote_code,
            truncation=True,
            truncation_side="left",  # truncate from left since we judge the completion
        )
    else:
        reward_model = None
        reward_tokenizer = None

    # initialize judge if specified
    if training_args.judge is not None:
        judge_cls = JUDGES[training_args.judge]
        judge = judge_cls()
    else:
        judge = None

    # set up tokenizer for main model
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",  # pad on left for generation
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE  # use default chat template
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token  # set pad token to eos token

    dataset = load_dataset(DATASET_NAME, name=DATASET_CONFIG)  # load training dataset

    # create online DPO trainer
    trainer = OnlineDPOTrainer(
        model=model,
        reward_model=reward_model,
        judge=judge,
        args=training_args,
        train_dataset=dataset[DATASET_TRAIN_SPLIT],
        eval_dataset=dataset[DATASET_TEST_SPLIT] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        reward_processing_class=reward_tokenizer,
        peft_config=get_peft_config(model_args) if USE_PEFT else None,  # configure LoRA if specified
    )

    # set up completion logging for evaluation
    if training_args.eval_strategy != "no":
        generation_config = GenerationConfig(
            max_new_tokens=training_args.max_new_tokens, do_sample=True, temperature=training_args.temperature
        )
        completions_callback = LogCompletionsCallback(trainer, generation_config, num_prompts=8)  # log 8 sample completions
        trainer.add_callback(completions_callback)

    trainer.train()  # start training

    # save trained model
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=DATASET_NAME)  # upload to hugging face hub