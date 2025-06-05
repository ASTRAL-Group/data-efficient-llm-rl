import os
import pickle
import torch
from transformers import AutoModel, AutoTokenizer
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--left_padding", action="store_true")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
parser.add_argument("--train_data", type=str, default="10k")
args = parser.parse_args()

# Initialize accelerator to handle device placement
accelerator = Accelerator()
device = accelerator.device
print(f"Using device: {device}")
print(f"Total GPUs available: {torch.cuda.device_count()}")

# Set paths
data_path = "datasets"
embedding_dir = os.path.join(data_path, "embeddings")
os.makedirs(embedding_dir, exist_ok=True)

# Load questions
print("Loading questions...")

file_name = f"{args.train_data}.parquet"
questions = pd.read_parquet(os.path.join(data_path, file_name))
questions = questions["problem"].tolist()
print(f"Loading model and tokenizer: {args.model_name}")
embedding_path = os.path.join(embedding_dir, f"{args.model_name.replace('/', '_')}_{args.train_data}.pt")
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token 
if args.left_padding:
    tokenizer.padding_side = "left" 
print(f"Padding side: {tokenizer.padding_side}")

# Load model with auto device map using accelerate
print("Loading model across available GPUs...")
if torch.cuda.device_count() > 1:
    # Method 1: Using auto device map
    model = AutoModel.from_pretrained(
        args.model_name,
        device_map="auto",  # Automatically distribute across available GPUs
        trust_remote_code=True,
        output_hidden_states=True
    )
    print(f"Model loaded with device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")
else:
    # Fallback to single GPU
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        output_hidden_states=True,
        attn_implementation="eager"
    ).to(device)
    print("Model loaded on single device")

# Set batch size based on available memory
batch_size = 2  # Start with a smaller batch size for large models like Qwen-7B

# Generate embeddings in batches
all_embeddings = []
print(f"Generating embeddings for {len(questions)} questions...")

for i in tqdm(range(0, len(questions), batch_size)):
    batch_questions = questions[i:i+batch_size]
    
    # Tokenize batch
    inputs = tokenizer(
        batch_questions, 
        padding=True, 
        truncation=True, 
        return_tensors="pt"
    )
    
    # Move inputs to the appropriate device
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    
    # Generate embeddings (without gradient calculation)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Get last hidden states from the model
    # For Qwen, we need to extract the hidden states differently than BERT
     # Debug the output structure
    attention_mask = inputs['attention_mask']
    if hasattr(outputs, 'last_hidden_state'):
        last_hidden_state = outputs.last_hidden_state
    elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        last_hidden_state = outputs.hidden_states[-1]
    else:
        # Fallback to the first element which is typically the last_hidden_state
        last_hidden_state = outputs[0]
    
    if 'bert' in args.model_name.lower():
        # Take CLS token 
        emb_output = last_hidden_state[:, 0, :]
    else:
        # Take average of all tokens
        expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        sum_hidden = (last_hidden_state * expanded_mask).sum(dim=1)
        emb_output = sum_hidden / expanded_mask.sum(dim=1) 
    
    all_embeddings.append(emb_output)
    
    # Print memory usage every 10 batches
    if i % 10 == 0:
        print(f"Memory usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

# Concatenate all batches
all_embeddings = torch.cat(all_embeddings, dim=0)
print(f"Embedding shape: {all_embeddings.shape}")

# Save embeddings as tensor file
torch.save(all_embeddings, embedding_path)
print(f"Embeddings saved to {embedding_path}")

print("Done!")

