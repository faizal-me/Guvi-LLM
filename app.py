from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import gradio as gr

# Load the fine-tuned model and tokenizer
model_name_or_path = "./fine_tuned_model"
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the text generation function
def generate_text(seed_text, max_length=100, temperature=1.0, num_return_sequences=1):
    # Tokenize the input text
    input_ids = tokenizer.encode(seed_text, return_tensors='pt').to(device)

    # Create attention mask
    attention_mask = torch.ones(input_ids.shape, device=device)

    # Generate text
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask
        )

    # Decode the generated text
    generated_texts = []
    for i in range(num_return_sequences):
        generated_text = tokenizer.decode(output[i], skip_special_tokens=True)
        generated_texts.append(generated_text)

    return generated_texts

# Define the function to be used by Gradio interface
def predict(seed_text, max_length, temperature, num_return_sequences):
    generated_texts = generate_text(seed_text, max_length, temperature, num_return_sequences)
    return "\n\n".join(generated_texts)

# Gradio interface definition
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Textbox(lines=2, placeholder="Enter seed text here...", label="Seed Text"),
        gr.Slider(minimum=50, maximum=500, value=50, step=1, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=1.5, value=1.0, step=0.1, label="Temperature"),
        gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Number of Return Sequences")
    ],
    outputs=gr.Textbox(),
    title="GPT-2 Text Generation",
    description="Enter some text and see the generated output based on the fine-tuned GPT-2 model."
)

# Launch the Gradio interface
if __name__ == "__main__":
    interface.launch()
