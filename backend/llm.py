import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()


# ---------------------------
# Device helpers
# ---------------------------
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = get_device()

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Mistral 7B Local Chat", layout="centered")
st.title("Mistral 7B Instruct (Local)")

# ---------------------------
# Load model (cached)
# ---------------------------
@st.cache_resource
def load_model():
    # model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token

    # quantization_config = None
    # if device == "cuda":
    #     quantization_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=torch.float16,
    #     )

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    #     quantization_config=quantization_config,
    # )
    

    # model.to(device)
    # model.eval()

    # if device == "mps":
    #     model.config.use_cache = False

    # pipe = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     max_new_tokens=512,
    #     temperature=0.2
    # )

    # llm = HuggingFacePipeline(pipeline=pipe)
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.1,
        max_tokens=512,
    )
    return None, llm 
    #return tokenizer, llm

tokenizer, model = load_model()
