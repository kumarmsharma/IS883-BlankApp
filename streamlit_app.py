import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

st.title("Simple GPT-2 Text Generator")

user_prompt = st.text_input("Enter your prompt here:")

num_tokens = st.number_input("How many tokens should the response be?", min_value=5, max_value=100, value=50)

if st.button("Generate Response"):
    if user_prompt:
        inputs = tokenizer.encode(user_prompt, return_tensors="pt")

        output_low = model.generate(inputs, max_length=num_tokens, do_sample=True, temperature=0.2)
        response_low = tokenizer.decode(output_low[0], skip_special_tokens=True)

        output_high = model.generate(inputs, max_length=num_tokens, do_sample=True, temperature=0.9)
        response_high = tokenizer.decode(output_high[0], skip_special_tokens=True)

        st.write("Low Creativity Response:")
        st.write(response_low)
        
        st.write("High Creativity Response:")
        st.write(response_high)
    else:
        st.write("Please enter a prompt!")
