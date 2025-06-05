from trl import OnlineDPOConfig, OnlineDPOTrainer
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from judge import CustomJudge  # my custom judge implementations

MODEL = "google/gemma-3-1b-it"

model = AutoModelForCausalLM.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
judge = CustomJudge()
train_dataset = load_dataset("json", data_files="synthetic_data_converted/combined_dataset.jsonl")
training_args = OnlineDPOConfig(output_dir="gemma-3-4b-dpo", logging_steps=10)
trainer = OnlineDPOTrainer(
    model=model, judge=judge, args=training_args, processing_class=tokenizer, train_dataset=train_dataset
)
trainer.train()