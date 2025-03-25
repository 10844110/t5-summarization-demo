import gradio as gr
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "M11309206/t5-summarization"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def summarize(text):
    input_text = "summarize: " + text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=100, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

demo = gr.Interface(
    fn=summarize,
    inputs=gr.Textbox(lines=10, placeholder="pls input ...", label="original content"),
    outputs=gr.Textbox(label="summary output"),
    title="🧠 T5 summary machine",
    description="pls input something, click and T5 will give u the summary!"
)

demo.launch()
