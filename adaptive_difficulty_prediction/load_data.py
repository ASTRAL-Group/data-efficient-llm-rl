import os
import pickle
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from collections import Counter

class TeacherDataset(object):
    def __init__(self, data_path, model_name):
        self.data_path = data_path
        self.model_name = model_name

        with open(os.path.join(data_path, "data_train.pkl"), "rb") as f:
            self.data_train = pickle.load(f)
        with open(os.path.join(data_path, "data_ref.pkl"), "rb") as f:
            self.data_ref = pickle.load(f)
        
    def load_embeddings(self):
        questions = pd.read_parquet(os.path.join(self.data_path, "questions.parquet"))['problem'].tolist()
        embeddings = torch.load(os.path.join(self.data_path, 'embeddings', f'{self.model_name.replace("/", "_")}.pt'))
       
        if len(questions) != embeddings.shape[0]:
            raise ValueError(f"Number of questions ({len(questions)}) does not match number of embeddings ({embeddings.shape[0]})")
        
        embeddings_dict = {questions[i]: embeddings[i] for i in range(len(questions))}
        print(f"Number of questions: {len(questions)}")
        print(f"Number of embeddings: {len(embeddings_dict)}")
        print(f"Number of duplications: {len(questions)-len(embeddings_dict)}")
        return embeddings_dict

    def load_train_data(self):
        train_questions = []
        train_rewards = []
        group_ids = []
        for model_name, item in self.data_train.items():
            train_questions_group = item['questions']
            train_rewards_group = item['rewards']
            train_questions.extend(train_questions_group)
            train_rewards.extend(train_rewards_group)
            group_ids.extend([model_name]*len(train_questions_group))
        assert len(train_rewards) == len(train_questions), "train_rewards and train_questions must have the same length"
        assert len(group_ids) == len(train_rewards), "group_ids and train_rewards must have the same length"
        
        ref_candidate_questions = [item["questions"] for model_name, item in self.data_ref.items()]
        for i in range(len(ref_candidate_questions)):
            assert ref_candidate_questions[0] == ref_candidate_questions[i], f"Questions in ref_data are not in the same order for all models"
        ref_candidate_questions=ref_candidate_questions[0]
        group_ref_candidate_labels = {model_name:item["rewards"] for model_name, item in self.data_ref.items()}
        return train_questions, train_rewards, group_ids, ref_candidate_questions, group_ref_candidate_labels

    def load_test_data(self,data,test_group_id,ref_size,seed=42):
        group_questions = data[test_group_id]['questions']
        group_rewards = data[test_group_id]['rewards']
        
        test_questions = []
        test_rewards = {}
        test_ref_questions = []
        test_ref_rewards = {}
        
        random.seed(seed)
        all_indices = list(range(len(group_questions)))
        random.shuffle(all_indices)
        
        test_ref_questions = [group_questions[i] for i in all_indices[:ref_size]]
        test_ref_rewards = [group_rewards[i] for i in all_indices[:ref_size]]
        
        test_questions = [group_questions[i] for i in all_indices[ref_size:]]
        test_rewards = [group_rewards[i] for i in all_indices[ref_size:]]
        test_group_ids = [test_group_id]*len(test_questions)
            
        return test_questions, test_rewards, test_group_ids, test_ref_questions, test_ref_rewards

    
class QuestionDataset(Dataset):
    def __init__(self, group_ids, questions, rewards):
        self.group_ids = group_ids
        self.questions = questions
        self.rewards = rewards

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.group_ids[idx], self.questions[idx], self.rewards[idx]
    
    
class QuestionEmbeddingDataset(Dataset):
    def __init__(self, group_ids, questions, rewards, embeddings_dict):
        self.group_ids = group_ids
        self.questions = questions
        self.rewards = rewards
        self.embeddings_dict = embeddings_dict  # Dictionary mapping question to its embedding

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions[idx]
        if isinstance(question, list):
            print(type(question), question)
        embedding = self.embeddings_dict.get(question, None)
        return self.group_ids[idx], question, self.rewards[idx], embedding
