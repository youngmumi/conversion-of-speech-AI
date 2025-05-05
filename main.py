!pip install transformers datasets

from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import os

# 👉 W&B 끄기 (원하지 않으면 생략 가능)
os.environ["WANDB_DISABLED"] = "true"

# 모델과 토크나이저 로딩
model = GPT2LMHeadModel.from_pretrained("skt/kogpt2-base-v2")
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token='</s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', mask_token='<mask>')

# 데이터 로딩
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="hannibal_lines.txt",  # 업로드한 파일명
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False,
)

# 학습 인자 설정
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

# Trainer 정의
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# 학습 실행
trainer.train()

# 모델 저장
trainer.save_model("./kogpt2_hannibal")
tokenizer.save_pretrained("./kogpt2_hannibal")