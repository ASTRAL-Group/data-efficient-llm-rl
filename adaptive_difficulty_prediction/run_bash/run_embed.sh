#!/bin/bash
EMBEDDING_MODEL_NAME="Qwen/Qwen2.5-Math-1.5B-Instruct"
train_data_mode="questions" 
python save_embedding.py --model_name $EMBEDDING_MODEL_NAME --left_padding --train_data $train_data_mode