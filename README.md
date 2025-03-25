# 📘 T5 Summarization Model

This is a fine-tuned [T5](https://huggingface.co/t5-small) model for text summarization tasks.

## 🧠 Model Info

- Base model: `t5-small`
- Task: Text Summarization
- Epochs: 3
- Training loss: 1.80
- Eval loss: 1.63
- ROUGE-1: 41.49
- ROUGE-2: 13.62
- ROUGE-L: 29.42

## 🚀 Usage (in Python)

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

model = T5ForConditionalGeneration.from_pretrained("你的帳號/t5-summarization")
tokenizer = T5Tokenizer.from_pretrained("你的帳號/t5-summarization")

text = "summarize: The Eiffel Tower is a famous landmark in Paris..."
inputs = tokenizer(text, return_tensors="pt", truncation=True)
outputs = model.generate(inputs.input_ids, max_length=100, num_beams=4, early_stopping=True)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(summary)

