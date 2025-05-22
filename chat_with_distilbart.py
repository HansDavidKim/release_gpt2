import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

save_dir   = "./distilbart_cnn_safetensors"
model_name = "sshleifer/distilbart-cnn-6-6"

if os.path.isdir(save_dir) and os.path.exists(os.path.join(save_dir, "pytorch_model.safetensors")):
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model     = AutoModelForSeq2SeqLM.from_pretrained(save_dir, from_safetensors=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir, safe_serialization=True)
    tokenizer.save_pretrained(save_dir)

model.eval()

def chat_with_bart(input_text, max_len=60):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_len,
        min_length=20,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("distilBART 챗봇 (종료하려면 'exit' 또는 'quit' 입력)")
while True:
    user_input = input("You: ")
    if user_input.strip().lower() in ("exit", "quit"):
        print("챗봇을 종료합니다.")
        break
    response = chat_with_bart(user_input)
    print("Bot:", response)
