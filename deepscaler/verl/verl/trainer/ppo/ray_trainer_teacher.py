# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# teacher
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

import numpy as np
from tqdm import tqdm
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
import shutil 
import time
from collections import defaultdict
import random

# from verl.trainer.ppo.teacher_utils import TeacherModel, load_embeddings

WorkerType = Type[Worker]

def dataprotoitem_to_dataproto(item: DataProtoItem) -> DataProto:
    """Convert a DataProtoItem to a DataProto object"""
    return DataProto.from_dict(
        tensors=item.batch,  # TensorDict is already in correct format
        non_tensors=item.non_tensor_batch,  # Dict is already in correct format 
        meta_info=item.meta_info
    )

class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6
    Teacher = 7

@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == 'gae':
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == 'grpo':
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_step_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }

def compute_epoch_metrics(epoch_raw):
    return {
        **{
            f'epoch/{name}': value for name, value in epoch_raw.items()
        },
    }

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.use_teacher = Role.Teacher in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        self._create_dataloader()

    def _create_dataloader(self):
        from torch.utils.data import DataLoader
        # TODO: we have to make sure the batch size is divisible by the dp size
        # from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
        # self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
        #                                  tokenizer=self.tokenizer,
        #                                  prompt_key=self.config.data.prompt_key,
        #                                  max_prompt_length=self.config.data.max_prompt_length,
        #                                  filter_prompts=True,
        #                                  return_raw_chat=self.config.data.get('return_raw_chat', False),
        #                                  truncation='error',
        #                                  format_reward=self.config.data.get('format_reward', False))
        # train_batch_size = self.config.data.train_batch_size
        # if self.config.trainer.rejection_sample:
        #     train_batch_size *= self.config.trainer.rejection_sample_multiplier
        #     train_batch_size = int(train_batch_size)
        # self.train_dataloader = DataLoader(dataset=self.train_dataset,
        #                                    batch_size=train_batch_size,
        #                                    shuffle=True, 
        #                                    drop_last=True,
        #                                    collate_fn=collate_fn)

        # self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
        #                                tokenizer=self.tokenizer,
        #                                prompt_key=self.config.data.prompt_key,
        #                                max_prompt_length=self.config.data.max_prompt_length,
        #                                filter_prompts=True,
        #                                return_raw_chat=self.config.data.get('return_raw_chat', False),
        #                                truncation='error',
        #                                  format_reward=self.config.data.get('format_reward', False))
        # self.val_dataloader = DataLoader(dataset=self.val_dataset,
        #                                  batch_size=len(self.val_dataset),
        #                                  shuffle=True,
        #                                  drop_last=True,
        #                                  collate_fn=collate_fn)

        # assert len(self.train_dataloader) >= 1
        # assert len(self.val_dataloader) >= 1

        # print(f'Size of train dataloader: {len(self.train_dataloader)}')
        # print(f'Size of val dataloader: {len(self.val_dataloader)}')

        # set total steps and save freq
        self.total_training_steps = self.config.data.mu * self.config.trainer.total_epochs
        if self.config.trainer.save_freq < 0:
            self.config.trainer.save_freq = self.config.data.mu

        # inject total_training_steps to actor/critic optim_config. This is hacky.

        if self.config.trainer.total_training_steps is not None:
            raise NotImplementedError

        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = self.total_training_steps
            self.config.critic.optim.total_training_steps = self.total_training_steps

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            # test_batch = test_batch.to('cuda')

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            n_val_samples = self.config.actor_rollout_ref.rollout.n_val
            test_batch = test_batch.repeat(repeat_times=n_val_samples, interleave=True)
            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_gen_batch_padded.meta_info['val_temperature'] = self.config.actor_rollout_ref.rollout.val_temperature
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('Validation: Generation end.')

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward
            reward_tensor = self.val_reward_fn(test_batch)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)

        return metric_dict

    def init_workers(self):
        self.prefix_name = "verl_controller"
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.config.algorithm.adv_estimator == 'gae':
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == 'grpo':
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # init teacher model
        if self.use_teacher:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Teacher)
            print(f"Teacher resource pool: {resource_pool}")
            teacher_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.Teacher],
                                                config=self.config.teacher_model,
                                            )
            self.resource_pool_to_cls[resource_pool]['teacher'] = teacher_cls
        
        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            print("[Worker Group] Creating worker group for other roles")
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, name_prefix=self.prefix_name)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)
                
        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # initialize teacher
        if self.use_teacher:
            self.teacher_wg = all_wg['teacher']
            self.teacher_wg.init_model()
        
        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self,save_checkpoints = False):
        if save_checkpoints:
            actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        'final_checkpoints')
            # latest checkpointed iteration tracker (for atomic usage)
            local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                            'latest_checkpointed_iteration.txt')
            with open(local_latest_checkpointed_iteration, 'w') as f:
                f.write(str(self.global_steps))
        else:
            actor_local_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                                        f'global_step_{self.global_steps}')
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path,save_checkpoints=save_checkpoints)     

        if self.use_critic:
            if save_checkpoints:
                critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                        'final_checkpoints')
            else:
                critic_local_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                                             f'global_step_{self.global_steps}')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, 'critic')
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path,save_checkpoints=save_checkpoints)
    
        # save dataloader
        # dataloader_local_path = os.path.join(self.config.trainer.default_local_dir, 'dataloader',
        #                                 f'global_step_{self.global_steps}','data.pt')
        # dataloader_state_dict = self.train_dataloader.state_dict()
        # torch.save(dataloader_state_dict, dataloader_local_path)

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        if self.config.trainer.resume_mode == 'auto':
            if self.config.trainer.resume_from_path is None:
                print('Training from scratch')
                return 0
             
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                   'latest_checkpointed_iteration.txt')
        if os.path.exists(local_latest_checkpointed_iteration):
            with open(local_latest_checkpointed_iteration, 'r') as f:
                self.global_steps = int(f.read().strip())
        else:
            print('No latest checkpoint found, training from scratch')
            return 0

        final_ckp_path = self.config.trainer.resume_from_path
        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {final_ckp_path}')

        actor_path = os.path.join(self.config.trainer.default_local_dir, 'actor',
                     self.config.trainer.resume_from_path)
        critic_path = os.path.join(self.config.trainer.default_local_dir, 'critic',
                     self.config.trainer.resume_from_path)
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=False) # hard code to False
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=False) # hard code to False

        # load dataloader,
        # TODO: from remote not implemented yet
        # dataloader_local_path = os.path.join(self.config.trainer.default_local_dir, 'dataloader',
        #              self.config.trainer.resume_from_path, 'data.pt')
        # if os.path.exists(dataloader_local_path):
        #     dataloader_state_dict = torch.load(dataloader_local_path)
        #     self.train_dataloader.load_state_dict(dataloader_state_dict)
        # else:
        #     print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")
        
    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf
        import traceback
        import sys
        import time
        
        print("Start to train!")

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0
        
        self._load_checkpoint()
        
        # load teacher model and embedding dict
        # self.embeddings_dict = load_embeddings(self.config.teacher_model.embedding_path, self.config.teacher_model.model_name)
        # self.teacher_model = TeacherModel(self.embedding_dict)
        # self.teacher_model = TeacherModel.remote(self.embeddings_dict)
        # resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)


        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        effective_total_steps = self.total_training_steps - self.global_steps
        trained_epochs = self.global_steps // self.config.data.mu
        
        # we start from step 1
        self.global_steps += 1
        
        # Maximum number of retry attempts
        max_retries = self.config.trainer.get('max_retries', 3)
        retry_count = 0
        retry_delay = self.config.trainer.get('retry_delay', 60)  # seconds
        
        while retry_count <= max_retries:
            try:
                with tqdm(total=effective_total_steps, desc="Training Progress") as pbar:

                    for epoch in range(trained_epochs, self.config.trainer.total_epochs):
                        epoch_raw = {}
                        epoch_metrics = {}

                        with _timer('epoch', epoch_raw):
                            with _timer('data_selection', epoch_raw):
                                from torch.utils.data import DataLoader
                                from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
                                self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                        tokenizer=self.tokenizer,
                                        prompt_key=self.config.data.prompt_key,
                                        max_prompt_length=self.config.data.max_prompt_length,
                                        filter_prompts=True,
                                        return_raw_chat=self.config.data.get('return_raw_chat', False),
                                        truncation='error',
                                        format_reward=self.config.data.get('format_reward', False)
                                        )
                                
                                if not self.config.data.random_selection:
                                    print("Not random!")
                                
                                    ref_indices = random.sample(range(len(self.train_dataset)), self.config.data.ref_size)
                                    ref_dataset = torch.utils.data.Subset(self.train_dataset, indices=ref_indices)  

                                    assert len(ref_dataset) % min(self.config.data.train_batch_size, self.config.data.ref_size) == 0
                                    ref_dataloader = DataLoader(dataset=ref_dataset,
                                            batch_size=min(self.config.data.train_batch_size, self.config.data.ref_size),
                                            shuffle=False, 
                                            drop_last=False,
                                            collate_fn=collate_fn)
                                    
                                    # ref data collection, labeling, save
                                    ref_batches = []
                                    ref_solve_none = 0
                                    ref_solve_all = 0
                                    for _, batch_dict in enumerate(tqdm(ref_dataloader, desc="Reference Rollout")):
                                        batch: DataProto = DataProto.from_single_dict(batch_dict)
                                        gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                                        batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                                dtype=object)
                                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                                        batch = batch.union(gen_batch_output)
                                        if self.use_critic:
                                            values = self.critic_wg.compute_values(batch)
                                            batch = batch.union(values)
                                        if self.use_rm:
                                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                                            batch = batch.union(reward_tensor)
                                        reward_tensor = self.reward_fn(batch)
                                        batch.batch['token_level_scores'] = reward_tensor
                                        uids = batch.non_tensor_batch['uid']
                                        unique_uids = np.unique(uids)                
                                        for uid in unique_uids:
                                            uid_mask = uids == uid
                                            uid_rewards = reward_tensor[uid_mask].sum(-1)  
                                            if (uid_rewards == 0).all():
                                                ref_solve_none += 1
                                            elif (uid_rewards == 1).all():
                                                ref_solve_all += 1
                                        ref_batches.append(batch)
                                        
                                    epoch_metrics['epoch/ref_solve_none'] = ref_solve_none
                                    epoch_metrics['epoch/ref_solve_all'] = ref_solve_all
                                    
                                    ref_batches = DataProto.concat(ref_batches)
                                    ref_data = defaultdict(dict)
                                    for _, batch in enumerate(ref_batches):
                                        key = batch.non_tensor_batch['index']
                                        ref_data[key]['question'] = batch.non_tensor_batch['extra_info']['question']
                                        ref_data[key]['rewards'] = ref_data[key].get('rewards', []) + [batch.batch['token_level_scores'].sum(-1).item()]
                                    
                                    ref_questions = []
                                    ref_labels = []
                                    ref_indices = []
                                    for key in ref_data:
                                        ref_questions.append(ref_data[key]['question'])
                                        ref_labels.append(np.mean(ref_data[key]['rewards']))
                                        ref_indices.append(key)
                                    
                                    import pickle
                                    ref_data_save_path = os.path.join(self.config.trainer.default_local_dir,'saved_ref_data')
                                    if not os.path.exists(ref_data_save_path):
                                        os.makedirs(ref_data_save_path)
                                    with open(os.path.join(ref_data_save_path,
                                                    f'ref_data_epoch_{epoch}.pkl'), 'wb') as f:
                                        pickle.dump(ref_data, f)
                                    print(f"[Ref] Saved {len(ref_data)} items to {ref_data_save_path}")

                                    # TODO: load teacher model, get predictions
                                    all_questions = [item['extra_info']['question'] for item in self.train_dataset]
                                    # now you get all_questions (list of texts), ref_indices, ref_questions (list of texts), ref_labels
                                    predicted_labels = self.teacher_wg.predict(all_questions, ref_questions, ref_labels, ref_indices, batch_size=self.config.teacher_model.batch_size)
                                    print("Teacher Prediction Success!")
                                else:
                                    predicted_labels = torch.rand(len(self.train_dataset))
                                    print("Random!")
                                    
                                predicted_label_save_path = os.path.join(self.config.trainer.default_local_dir, 'saved_predicted_labels')
                                if not os.path.exists(predicted_label_save_path):
                                    os.makedirs(predicted_label_save_path)
                                torch.save(predicted_labels, os.path.join(predicted_label_save_path, f'predicted_labels_epoch_{epoch}.pt'))
                                print(f"[Predicted Labels] Saved {len(predicted_labels)} labels to {predicted_label_save_path}")

                                # get selected subset
                                selection_budget = int(self.config.data.mu * self.config.data.train_batch_size)
                                dataset_sampling_scores = -torch.abs(predicted_labels - self.config.data.alpha)
                                # softmax
                                dataset_sampling_logits = dataset_sampling_scores / self.config.data.tau
                                dataset_sampling_logits -= dataset_sampling_logits.max()
                                dataset_sampling_probabilities = torch.softmax(dataset_sampling_logits, dim=0)
                                selected_indices = torch.multinomial(dataset_sampling_probabilities, selection_budget, replacement=False)

                                # greedy
                                # if self.config.data.tau == "greedy":
                                #     selected_indices = torch.topk(dataset_sampling_scores, selection_budget).indices
                                # elif self.config.data.tau.startswith("gumbel-"):
                                #     temp_val = float(self.config.data.tau.split("-")[1])
                                #     gumbel_noise = -torch.log(-torch.log(torch.rand_like(dataset_sampling_scores)))
                                #     dataset_sampling_logits = dataset_sampling_scores / temp_val + gumbel_noise
                                #     selected_indices = torch.topk(dataset_sampling_logits, selection_budget).indices
                                # else:
                                #     raise

                                selected_indices = selected_indices[torch.randperm(len(selected_indices))] # shuffle

                                selected_indices_save_path = os.path.join(self.config.trainer.default_local_dir, 'saved_selected_indices')
                                if not os.path.exists(selected_indices_save_path):
                                    os.makedirs(selected_indices_save_path)
                                torch.save(selected_indices, os.path.join(selected_indices_save_path, f'selected_indices_epoch_{epoch}.pt'))
                                print(f"[Selected Indices] Saved {len(selected_indices)} indices to {selected_indices_save_path}")

                                use_dataset = torch.utils.data.Subset(self.train_dataset, indices=selected_indices.tolist())  
                                assert len(use_dataset) % self.config.data.train_batch_size == 0
                                use_train_dataloader = DataLoader(dataset=use_dataset,
                                                            batch_size=self.config.data.train_batch_size,
                                                            shuffle=False, 
                                                            drop_last=False, 
                                                            collate_fn=collate_fn)
                                assert len(use_train_dataloader) == self.config.data.mu

                                
                            with _timer('Rollout_update', epoch_raw):
                                for _, batch_dict in enumerate(tqdm(use_train_dataloader, desc="Rollout_update")):
                                    batch: DataProto = DataProto.from_single_dict(batch_dict)

                                    metrics = {}
                                    
                                    # pop those keys for generation
                                    gen_batch = batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])

                                    # generate a batch
                                    gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                                    # This code matches a prompt ID with its N responses.
                                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                                            dtype=object)
                                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                                    batch = batch.union(gen_batch_output)

                                    # compute values
                                    if self.use_critic:
                                        values = self.critic_wg.compute_values(batch)
                                        batch = batch.union(values)

                                    # compute scores using reward model and/or reward function
                                    if self.use_rm:
                                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                                        batch = batch.union(reward_tensor)

                                    reward_tensor = self.reward_fn(batch)
                                    batch.batch['token_level_scores'] = reward_tensor

                                    uids = batch.non_tensor_batch['uid']
                                    unique_uids = np.unique(uids)      
                                    solve_none = 0
                                    solve_all = 0          

                                    for uid in unique_uids:
                                        uid_mask = uids == uid
                                        uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence
                                        
                                        # Check if all rewards are 0 or all are 1 for this uid
                                        if (uid_rewards == 0).all():
                                            solve_none += 1
                                        elif (uid_rewards == 1).all():
                                            solve_all += 1

                                    metrics['batch/solve_none'] = solve_none
                                    metrics['batch/solve_all'] = solve_all

                                    saved_data = defaultdict(dict)
                                    for _, single_batch in enumerate(batch):
                                        key = single_batch.non_tensor_batch['index']
                                        response_token_ids = single_batch.batch['responses']
                                        response_texts = self.tokenizer.decode(response_token_ids, skip_special_tokens=True)
                                        saved_data[key]['question'] = single_batch.non_tensor_batch['extra_info']['question']
                                        saved_data[key]['responses'] = saved_data[key].get('responses', []) + [response_texts]
                                        saved_data[key]['rewards'] = saved_data[key].get('rewards', []) + [single_batch.batch['token_level_scores'].sum(-1).item()]
                                    
                                    import pickle
                                    data_save_path = os.path.join(self.config.trainer.default_local_dir,'saved_data')
                                    if not os.path.exists(data_save_path):
                                        os.makedirs(data_save_path)
                                    with open(os.path.join(data_save_path,
                                                    f'rollout_data_step_{self.global_steps}.pkl'), 'wb') as f:
                                        pickle.dump(saved_data, f)

                                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                    batch = batch.union(old_log_prob)

                                    if self.use_reference_policy:
                                        # compute reference log_prob
                                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                        batch = batch.union(ref_log_prob)

                                    # compute rewards with KL penalty if needed

                                    # Note: This kl penalty applied directly over the rewards is disabled for GRPO. The kl penalty is applied at dp_actor.py
                                    # where it is subtracted directly from the policy loss

                                    # if not self.config.actor_rollout_ref.actor.use_kl_loss:
                                    #     batch, kl_metrics = apply_kl_penalty(batch,
                                    #                                        kl_ctrl=self.kl_ctrl,
                                    #                                        kl_penalty=self.config.algorithm.kl_penalty)
                                    #     metrics.update(kl_metrics)
                                    # else:
                                    #     batch.batch['token_level_rewards'] = batch.batch['token_level_scores']


                                    batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                                    # compute advantages, executed on the driver process
                                    batch = compute_advantage(batch,
                                                            adv_estimator=self.config.algorithm.adv_estimator,
                                                            gamma=self.config.algorithm.gamma,
                                                            lam=self.config.algorithm.lam,
                                                            num_repeat=self.config.actor_rollout_ref.rollout.n)

                                    # balance the number of valid tokens on each dp rank.
                                    # Note that this breaks the order of data inside the batch.
                                    # Please take care when you implement group based adv computation such as GRPO and rloo
                                    self._balance_batch(batch, metrics=metrics)

                                    # compute global_valid tokens
                                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                                    # update critic
                                    if self.use_critic:
                                        critic_output = self.critic_wg.update_critic(batch)
                                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                                        metrics.update(critic_output_metrics)

                                    # implement critic warmup
                                    if self.config.trainer.critic_warmup <= self.global_steps:
                                        # update actor
                                        actor_output = self.actor_rollout_wg.update_actor(batch)
                                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                                        metrics.update(actor_output_metrics)

                                    # validate
                                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                                        self.global_steps % self.config.trainer.test_freq == 0:
                                        val_metrics: dict = self._validate()
                                        metrics.update(val_metrics)

                                    if self.config.trainer.save_freq > 0 and \
                                            self.global_steps % self.config.trainer.save_freq == 0:
                                        print("Saving model...")
                                        with _timer('save_checkpoint', epoch_raw):
                                            self._save_checkpoint(save_checkpoints=False)
                                    
                                    if self.config.trainer.save_freq > 0 and \
                                            self.global_steps % (self.config.trainer.save_freq * 3) == 0:
                                        print("Saving model...")
                                        with _timer('save_checkpoint', epoch_raw):
                                            self._save_checkpoint(save_checkpoints=True)

                                    # collect metrics
                                    metrics.update(compute_step_metrics(batch=batch, use_critic=self.use_critic))

                                    # --- new code: decode prompts and responses ---
                                    prompt_token_ids = batch.batch['prompts']
                                    response_token_ids = batch.batch['responses']
                                    rewards = batch.batch['token_level_rewards'].max(-1).values

                                    prompt_texts = self.tokenizer.batch_decode(prompt_token_ids, skip_special_tokens=True)
                                    response_texts = self.tokenizer.batch_decode(response_token_ids, skip_special_tokens=True)

                                    # Optionally store only a small sample to avoid huge logs
                                    max_examples_to_log = 30
                                    sample_indices = range(min(len(prompt_texts), max_examples_to_log))
                                    logged_prompts = []
                                    logged_responses = []
                                    logged_rewards = []

                                    for i in sample_indices:
                                        logged_prompts.append(prompt_texts[i])
                                        logged_responses.append(response_texts[i])
                                        logged_rewards.append(rewards[i].item())

                                    if 'wandb' in logger.logger.keys():
                                        import pandas as pd
                                        import wandb
                                        table = {
                                                    "step": [str(self.global_steps)] * len(logged_prompts),
                                                    "prompt": logged_prompts,
                                                    "completion": logged_responses,
                                                    "reward": logged_rewards,
                                                }
                                        df = pd.DataFrame(table)
                                        logger.log(data={"completions": wandb.Table(dataframe=df)}, step=self.global_steps, backend='wandb')

                                    logger.log(data=metrics, step=self.global_steps)

                                    self.global_steps += 1
                                    pbar.update(1)
                                    
                            epoch_metrics.update(compute_epoch_metrics(epoch_raw=epoch_raw))
                            logger.log(data=epoch_metrics, step=self.global_steps-1)

                            # we dont need val for now
                            
                            if self.global_steps > self.total_training_steps:
                                self.global_steps -= 1
                                with _timer('save_final_checkpoint', epoch_raw):
                                    self._save_checkpoint(save_checkpoints=True)
                                print("Training completed successfully!")
                                return

                            #     # perform validation after training
                            #     if self.val_reward_fn is not None:
                            #         val_metrics = self._validate()
                            #         pprint(f'Final validation metrics: {val_metrics}')
                            #         logger.log(data=val_metrics, step=self.global_steps)
                            #     return
                                
                # If we reach here, training completed successfully
                self.global_steps -= 1
                with _timer('save_final_checkpoint', epoch_raw):
                    self._save_checkpoint(save_checkpoints=True)
                print("Training completed successfully!")

                epoch_metrics.update(compute_epoch_metrics(epoch_raw=epoch_raw))
                logger.log(data=epoch_metrics, step=self.global_steps)
                return
                
            except Exception as e:
                retry_count += 1
                error_msg = f"Error during training: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)
                logger.log(data={"error": error_msg}, step=self.global_steps)
                
                # Save the current state before exiting
                print(f"Saving emergency checkpoint at step {self.global_steps}")
                self._save_checkpoint(save_checkpoints=True)
                
                if retry_count <= max_retries:
                    print(f"Attempting to resume training in {retry_delay} seconds (attempt {retry_count}/{max_retries})")
                    time.sleep(retry_delay)
                    
                    # Reload the checkpoint to resume from where we left off
                    self._load_checkpoint()
                    print(f"Resumed training from step {self.global_steps}")
                else:
                    print(f"Maximum retry attempts ({max_retries}) reached. Exiting.")
                    raise
