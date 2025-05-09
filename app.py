import gradio as gr
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import mysql.connector
from mysql.connector import Error

# Load the fine-tuned model and tokenizer from Hugging Face Hub
model_name_or_path = "faizal-me/guvillm"
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Database connection setup
def create_connection():
    try:
        conn = mysql.connector.connect(
            host="gateway01.ap-southeast-1.prod.aws.tidbcloud.com",
            port=4000,
            user="4AZa7qXAXt5wnXK.root",
            password="3NDymp2pEOnxtdwV",
            database="test"
        )
        return conn
    except Error as e:
        print(f"Error: {e}")
        return None

conn = create_connection()
if conn:
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(255) NOT NULL,
            login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()

def generate_text(seed_text, max_length=100, temperature=1.0, num_return_sequences=1):
    input_ids = tokenizer.encode(seed_text, return_tensors='pt').to(device)
    attention_mask = torch.ones(input_ids.shape, device=device)
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
    generated_texts = [tokenizer.decode(output[i], skip_special_tokens=True) for i in range(num_return_sequences)]
    return generated_texts

def predict(seed_text, max_length, temperature, num_return_sequences):
    return "\n\n".join(generate_text(seed_text, max_length, temperature, num_return_sequences))

def check_registration(username):
    try:
        conn = create_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            return cursor.fetchone() is not None
    except Error as e:
        return False
    finally:
        if conn:
            conn.close()

def register_user(username):
    if check_registration(username):
        return "**Username already registered!**"
    
    try:
        conn = create_connection()
        if conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (username) VALUES (%s)", (username,))
            conn.commit()
            return "**User registered successfully!**"
    except Error as e:
        return f"**Error: {e}**"
    finally:
        if conn:
            conn.close()

def generate_text_tab(username, seed_text, max_length, temperature, num_return_sequences):
    if not check_registration(username):
        return "**You must register before accessing text generation.**"
    return f"**{predict(seed_text, max_length, temperature, num_return_sequences)}**"

with gr.Blocks() as demo:
    with gr.Row():
        gr.HTML("<h1 style='color: #4CAF50;'>Deployment GUVI GPT Model using Hugging Face</h1>")

    with gr.Tab("🏠 Home"):
        with gr.Row():
            with gr.Column():
                gr.HTML("<h2>Welcome to GUVI LLM! 🌟</h2>")
                gr.HTML("<img src='https://img-cdn.thepublive.com/fit-in/1200x675/entrackr/media/post_attachments/wp-content/uploads/2022/09/Guvi.jpg' />")
            
            with gr.Column():
                gr.Markdown("# 🚀 WHAT YOU GET FROM THIS") 
                recent_text = gr.Markdown(
                    "## 🎉 Experience the power of advanced text generation with our cutting-edge Large Language Model. "
                    "Transform your ideas into high-quality, insightful content with GUVI LLM. Whether you need creative writing, "
                    "technical explanations, or unique content tailored to your specifications, GUVI LLM is here to bring your text generation needs to life. 🌟"
                ) 

    with gr.Tab("📝 User Registration"):
        with gr.Column():
            gr.Markdown("### 📋 Register a New User")  
            username_reg = gr.Textbox(label="Username", placeholder="Enter a unique username...", lines=1)
            register_button = gr.Button("Register")
            register_output = gr.Markdown()
            register_button.click(register_user, inputs=[username_reg], outputs=register_output)

    with gr.Tab("✍️ Text Generation"):
        with gr.Column():
            gr.Markdown("### 📝 Generate Text") 
            username_gen = gr.Textbox(label="Username", placeholder="Enter your username...", lines=1)
            seed_text = gr.Textbox(lines=2, placeholder="Enter seed text here...", label="Seed Text")
            max_length = gr.Slider(minimum=50, maximum=500, value=100, step=10, label="Max Length")
            temperature = gr.Slider(minimum=0.1, maximum=1.5, value=1.0, step=0.1, label="Temperature")
            num_return_sequences = gr.Slider(minimum=1, maximum=5, value=1, step=1, label="Number of Return Sequences")
            generate_button = gr.Button("Generate")
            results_label = gr.Markdown("**Results:**")
            results = gr.Markdown()
            generate_button.click(lambda username, seed_text, max_length, temperature, num_return_sequences: generate_text_tab(username, seed_text, max_length, temperature, num_return_sequences), 
                                  inputs=[username_gen, seed_text, max_length, temperature, num_return_sequences], 
                                  outputs=results)

# Launch the Gradio interface
demo.launch(share=True)
