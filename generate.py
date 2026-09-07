import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

MODEL_DIR = "model/gpt2-custom"

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

prompt = input("Enter a prompt: ").strip()

if not prompt:
    prompt = "Artificial intelligence"

inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
    )

text = tokenizer.decode(output[0], skip_special_tokens=True)

print("\n--- Generated Text ---")
print(text)
