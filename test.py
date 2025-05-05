from transformers import pipeline, GPT2LMHeadModel, PreTrainedTokenizerFast

model = GPT2LMHeadModel.from_pretrained("./kogpt2_hannibal")
tokenizer = PreTrainedTokenizerFast.from_pretrained("./kogpt2_hannibal")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = "사랑해요"
result = generator(prompt, max_length=100, do_sample=True, top_k=50)

print("\n[한니발 스타일 생성 결과]")
print(result[0]["generated_text"])