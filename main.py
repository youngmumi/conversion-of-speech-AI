!pip install transformers datasets --upgrade

from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
import os

# W&B 비활성화
os.environ["WANDB_DISABLED"] = "true"

# 모델과 토크나이저
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
tokenizer = PreTrainedTokenizerFast.from_pretrained(
    "skt/kogpt2-base-v2",
    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
    pad_token='<pad>', mask_token='<mask>'
)

# 데이터 로딩
dataset = load_dataset("text", data_files={"train": "hannibal_lines.txt"})
def tokenize(example):
    return tokenizer(example["text"], return_special_tokens_mask=True)

tokenized = dataset.map(tokenize, batched=True)

# 데이터 콜레이터
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./kogpt2_hannibal",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_steps=100
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized["train"]
)

# 학습 및 저장
trainer.train()
trainer.save_model("./kogpt2_hannibal")
tokenizer.save_pretrained("./kogpt2_hannibal")
