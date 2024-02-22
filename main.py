import torch
import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load MoE-LLaVA Model
model_path = "path_to_your_model_directory"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# Function to generate text
def generate_text(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text

# Create Gradio Interface
iface = gr.Interface(fn=generate_text, inputs="text", outputs="text")
iface.launch()