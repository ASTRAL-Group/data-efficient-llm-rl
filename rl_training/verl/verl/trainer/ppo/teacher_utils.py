import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '8'
from torch import nn
import torch
import torch.nn.functional as F
from transformers import AutoModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm
from verl.single_controller.base import Worker
# from vllm import LLM
from verl.single_controller.base.decorator import Dispatch, collect_all_to_all, register, Execute

def build_projection(input_size, hidden_size, num_layers=1, use_layernorm=True, dropout=0.1):
    layers = []

    if use_layernorm:
        layers.append(nn.LayerNorm(input_size))

    for i in range(num_layers):
        if i == 0:
            layers.append(nn.Linear(input_size, hidden_size))
        else:
            layers.append(nn.Linear(hidden_size, hidden_size))
        if num_layers > 1 and i < num_layers - 1:
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
    
    layers.append(nn.LayerNorm(hidden_size))  

    return nn.Sequential(*layers)


class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased', lora=False):
        super().__init__()
        self.model_name = model_name
        self.encoder = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=True,  # Ensure hidden states are always available
            # torch_dtype=torch.float16 
        )
        if lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],  # You may need to inspect DistilBERT architecture; adjust if needed
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.encoder = get_peft_model(self.encoder, lora_config)
        self.hidden_size = self.encoder.config.hidden_size
        print(f"Hidden size: {self.hidden_size}")
        

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Check if token_type_ids is provided
        if token_type_ids is not None:
            outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Debug the output structure
        if hasattr(outputs, 'last_hidden_state'):
            last_hidden_state = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
            last_hidden_state = outputs.hidden_states[-1]
        else:
            # Fallback to the first element which is typically the last_hidden_state
            last_hidden_state = outputs[0]
            
        if 'bert' in self.model_name.lower():
            # Take CLS token 
            emb_output = last_hidden_state[:, 0, :]
        else:
            # Take average of all tokens
            expanded_mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            sum_hidden = (last_hidden_state * expanded_mask).sum(dim=1)
            emb_output = sum_hidden / expanded_mask.sum(dim=1) 
        
        return emb_output
    
class ResidualHead(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, scaling = 'platt', top_k=None):
        super().__init__()
        self.sim_head = RegressionHead(input_size, hidden_size, num_layers=num_layers, top_k=top_k)
        self.scaling = scaling
        if self.scaling == 'platt':
            self.res_scale = nn.Parameter(torch.ones(1))
            self.scale = nn.Parameter(torch.ones(1))
        elif self.scaling == 'temperature':
            self.scale = nn.Parameter(torch.ones(1))    
        elif self.scaling == 'group_logit_temp':
            self.mlp        = nn.Sequential(                
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            )
        elif self.scaling == 'plain':
            pass
        else:
            raise ValueError(f"Invalid scaling method: {scaling}")

    def forward(self, q, r, ref_vals,tau=1.0):
        base = self.sim_head(q, r, ref_vals, tau)             
        if self.scaling == 'platt':
            out  = torch.sigmoid(self.scale * torch.logit(base.clamp(1e-4, 1-1e-4)) + self.res_scale)
        elif self.scaling == 'temperature':
            out = torch.sigmoid(torch.logit(base.clamp(1e-4, 1-1e-4))/self.scale)
        elif self.scaling == 'group_logit_temp':
            mean_vec = torch.mean(ref_vals, dim=-1, keepdim=True)
            std_vec = torch.std(ref_vals, dim=-1, keepdim=True)
            concat_vec = torch.cat((mean_vec, std_vec), dim=-1)  # (B, 2)
            temp_bias = self.mlp(concat_vec)                     # (B, 2)
            temp = F.softplus(temp_bias[:, 0])                   # (B,)
            bias = torch.tanh(temp_bias[:, 1])                   # (B,)
            out = torch.sigmoid(torch.logit(base.clamp(1e-4, 1-1e-4)) / temp.clamp(1e-2, 10) + bias)
        elif self.scaling == 'plain':
            out = base
        return out
    

class RegressionHead(nn.Module):
    def __init__(self, input_size,hidden_size,num_layers=1,top_k=None):
        super().__init__()
        self.top_k = top_k
        self.query_proj = build_projection(input_size, hidden_size, num_layers=num_layers)
        self.ref_proj = build_projection(input_size, hidden_size, num_layers=num_layers)    

    def forward(self, query_repr, ref_repr, ref_values,tau=1.0):
        q_proj = self.query_proj(query_repr)
        r_proj = self.ref_proj(ref_repr)
        scores = torch.bmm(r_proj, q_proj.unsqueeze(-1)).squeeze(-1)/r_proj.size(-1) ** 0.5
        
        if self.top_k is not None and self.top_k < scores.size(1):
            k = self.top_k
            vals, idx = torch.topk(scores, k, dim=1)
            mask = scores.new_full(scores.shape, float('-inf'))
            mask.scatter_(1, idx, vals)
            scores = mask     

        weights = scores/tau
        weights -= torch.max(weights, dim=-1, keepdim=True).values
        weights = F.softmax(weights, dim=-1)                         # (B, K)
        pred    = (weights * ref_values).sum(dim=-1)                # (B,)
        return pred

class FewShotRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3,scaling='group_logit_temp', top_k=None):
        super().__init__()
        
        self.regressor = ResidualHead(input_size, hidden_size,num_layers,scaling=scaling, top_k=top_k)


    def forward(self, query_input, ref_input, ref_values):
        q_repr = query_input
        r_repr = ref_input

        B = q_repr.size(0)
        K = ref_values.size(1)
        r_repr = r_repr.unsqueeze(0).expand(B, K, -1)
                    
        return self.regressor(q_repr, r_repr, ref_values)

class QuestionDataset(Dataset):
    def __init__(self, questions):
        self.questions = questions

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx]
    
    
class QuestionEmbeddingDataset(Dataset):
    def __init__(self, questions, embeddings_dict):
        self.questions = questions
        self.embeddings_dict = embeddings_dict  # Dictionary mapping question to its embedding

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        if question not in self.embeddings_dict:
            raise
        return question, self.embeddings_dict.get(question, None)

def load_embeddings(embedding_path, model_name):
    questions = pd.read_parquet(os.path.join(embedding_path, 'questions', "questions.parquet"))['problem'].tolist()
    embeddings = torch.load(os.path.join(embedding_path, 'embeddings', f'{model_name.replace("/", "_")}.pt'), map_location=torch.device('cpu'))
    
    if len(questions) != embeddings.shape[0]:
        raise ValueError(f"Number of questions ({len(questions)}) does not match number of embeddings ({embeddings.shape[0]})")
    
    embeddings_dict = {questions[i]: embeddings[i] for i in range(len(questions))}
    print(f"Number of questions: {len(questions)}")
    print(f"Number of embeddings: {len(embeddings_dict)}")
    print(f"Number of duplications: {len(questions)-len(embeddings_dict)}")
    return embeddings_dict
 

class TeacherModelWorker(Worker):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings_dict = load_embeddings(config.embedding_path, config.model_name)

    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    def init_model(self):
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(self.config.model_name)
        self.input_size = config.hidden_size
        state_dict = torch.load(self.config.checkpoint_path, map_location=torch.device('cpu'))
        regressor_state_dict = {k: v for k, v in state_dict['model_state_dict'].items() if k.startswith('regressor.')}
        model = FewShotRegressor(input_size=self.input_size, hidden_size=self.config.hidden_size, num_layers=self.config.num_layers,scaling=self.config.scaling, top_k=self.config.top_k) 
        model.load_state_dict(regressor_state_dict)
        self.teacher_model = model.to("cuda")
        self.teacher_model.eval()

    @register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.RANK_ZERO)
    def predict(self, all_questions, ref_questions, ref_labels, ref_indices, batch_size=32):
        device = next(self.teacher_model.parameters()).device
        print(f"Predicting on {device}")
        query_dataset = QuestionEmbeddingDataset(all_questions, self.embeddings_dict)
        query_dataloader = DataLoader(
            query_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
        predicted_labels = []
        for batch in tqdm(query_dataloader, desc="Teacher model inference"):
            _, query_embeddings = batch
            ref_values_tensor = torch.tensor(
                [ref_labels], 
                dtype=torch.float32
            ).to(device)
            
            query_embeddings = torch.stack([emb.to(device) for emb in query_embeddings])
            ref_embeddings = torch.stack([self.embeddings_dict[ref_text].to(device)
                        for ref_text in ref_questions])
            with torch.no_grad():
                preds = self.teacher_model(query_embeddings, ref_embeddings, ref_values_tensor)
            predicted_labels.append(preds.cpu())

        # # calibration
        predicted_labels = torch.cat(predicted_labels, dim=0)
        import numpy as np
        X = np.array([predicted_labels[i] for i in ref_indices]).reshape(-1, 1)
        y = np.array(ref_labels).reshape(-1, 1)
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression()
        reg.fit(X, y)

        slope = reg.coef_[0][0]
        intercept = reg.intercept_[0]
        print(f"Linear transformation: y = {slope:.4f}x + {intercept:.4f}")
        
        final_predicted_labels = reg.predict(np.array(predicted_labels).reshape(-1, 1)).reshape(-1)
        
        # final_predicted_labels = torch.cat(predicted_labels, dim=0)
        final_predicted_labels = torch.tensor(final_predicted_labels, dtype=torch.float32)
        return final_predicted_labels
    