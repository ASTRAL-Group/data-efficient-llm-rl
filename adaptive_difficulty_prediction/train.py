import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
import argparse
import wandb 
from accelerate import Accelerator
from utils import pinball_loss, log_main_process, get_tqdm_iterator, interval_classification_metrics, focal_binary_cross_entropy, calibrate_predictions
from model import FewShotRegressor
from load_data import (
    QuestionDataset,
    QuestionEmbeddingDataset,
    TeacherDataset,
    RECORD_TEST_GROUPS
)
from torch.utils.data import DataLoader


CRITERION = {
    "mse": nn.MSELoss(),
    "binary_cross_entropy": nn.BCELoss(),
}

# Initialize metrics storage
metrics_for_plotting = {"test_losses": []}



def train_full_epoch(model, train_dataloader, ref_questions, ref_values, 
                     optimizer, accelerator, teacher_data_loader, embeddings_dict=None, 
                     epochs=3, logger=None, save_dir=None, loss_type="mse", ref_size=128,
                     scheduler=None,taus=None,tau=1.0, calibrator='linear'):
    model.train()
    metrics = {"epoch_losses": [], "test_losses": []}
    criterion = CRITERION[loss_type]
    
    # Cache reference texts processing to avoid repeating it for every batch
    batch_ref_texts = ref_questions
    
    # Create a progress bar for total training time
    total_pbar = tqdm(total=epochs, desc="Total Training Progress", position=0, disable=not accelerator.is_main_process)
    
    # Prepare reference encodings once if not using embeddings
    if not embeddings_dict:
        ref_enc = None
    
    for epoch in range(epochs):
        total_loss = 0.0
        batch_losses = []
        model.train()
        
        # Use the utility function for tqdm
        for log_step, batch_data in enumerate(get_tqdm_iterator(accelerator, train_dataloader, desc=f"Epoch {epoch+1}", position=1)):
            with accelerator.accumulate(model):
                has_embeddings = len(batch_data) == 4
                
                if has_embeddings:  
                    group_ids, query_texts, query_values, query_embeddings = batch_data
                else:  
                    group_ids, query_texts, query_values = batch_data
                
                if isinstance(group_ids, torch.Tensor):
                    group_ids = group_ids.cpu().tolist()
                
                # Sample reference indices for the batch
                ref_indices = random.sample(range(len(ref_questions)), k=ref_size)
                batch_ref_texts = [ref_questions[i] for i in ref_indices]
                
                # Prepare reference values for each group in the batch
                batch_ref_values = [
                    [ref_values[group_id][i] for i in ref_indices]
                    for group_id in group_ids
                ]
                ref_values_tensor = torch.tensor(
                    batch_ref_values, 
                    dtype=torch.float32
                ).to(accelerator.device)
                
                query_values = torch.tensor(
                    query_values, 
                    dtype=torch.float32
                ).to(accelerator.device)
                
                # Forward pass
                if has_embeddings and query_embeddings[0] is not None:
                    # Using pre-computed embeddings
                    query_embeddings = torch.stack([emb.to(accelerator.device) for emb in query_embeddings])
                    ref_embeddings = torch.stack([embeddings_dict[ref_text].to(accelerator.device) 
                                                for ref_text in batch_ref_texts])
                    preds = model(query_embeddings, ref_embeddings, ref_values_tensor, tau=tau)
                else:
                    # Using on-the-fly tokenization
                    query_enc = tokenizer(
                        query_texts, 
                        padding=True, 
                        truncation=True, 
                        return_tensors="pt",
                    ).to(accelerator.device)
                    
                    # Process references for the batch
                    ref_enc = tokenizer(
                        batch_ref_texts,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    ).to(accelerator.device)
                    
                    preds = model(query_enc, ref_enc, ref_values_tensor, tau=tau)
                
                # Calculate loss and backpropagate
                if loss_type == 'pinball':
                    loss = criterion(preds, query_values, tau=[0.1,0.5,0.9])
                else:
                    loss = criterion(preds, query_values)
                accelerator.backward(loss)
                
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

                # Calculate and log batch loss
                batch_loss = loss.item()
                batch_loss = torch.tensor(batch_loss, device=accelerator.device)
                all_batch_loss = accelerator.gather(batch_loss)
                mean_batch_loss = all_batch_loss.cpu().mean().item()
                
                batch_losses.append(mean_batch_loss)
                total_loss += mean_batch_loss
                
                # Log batch results
                log_msg = (f"Epoch {epoch+1}, Step {log_step+1}, "
                            f"Batch Loss: {mean_batch_loss:.4f}")
                log_main_process(accelerator, logger, log_msg)
                
                if accelerator.is_main_process and wandb.run is not None:
                    wandb.log({
                        "batch_loss": mean_batch_loss,
                        "step": epoch * len(train_dataloader) + log_step,
                        "learning_rate": optimizer.param_groups[0]['lr'] if scheduler else optimizer.param_groups[0]['lr']
                    },step=epoch * len(train_dataloader) + log_step)
        
        # Calculate and log epoch metrics
        num_batches = len(train_dataloader)
        epoch_loss = total_loss / num_batches
        metrics["epoch_losses"].append(epoch_loss)
        log_msg = f"Epoch {epoch+1} Total Loss: {epoch_loss:.4f}"
        log_main_process(accelerator, logger, log_msg)
        
        if accelerator.is_main_process and wandb.run is not None:
            wandb.log({"epoch_loss": epoch_loss, "epoch": epoch + 1})
            
        # Test evaluation on various test sets
        if epoch % 2 == 0:
            evaluate_test_groups(model, tokenizer, accelerator, logger, save_dir, epoch, data_path,
                teacher_data_loader, embeddings_dict, ref_size, calibrator, tau) 
            
            # Update metrics with test losses
            if hasattr(globals(), 'metrics_for_plotting'):
                metrics["test_losses"] = metrics_for_plotting["test_losses"]
        
        # Update the total progress bar
        total_pbar.update(1)
    
    # Close the total progress bar
    total_pbar.close()
    
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_model_path = os.path.join(save_dir, "model_final.pt")
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save({
            'model_state_dict': unwrapped_model.state_dict(),
            'config': config,
            'train_metrics': metrics,
        }, final_model_path)
        log_main_process(accelerator, logger, f"Final model saved to {final_model_path}")
    
    return metrics


def evaluate_test_groups(model, tokenizer, accelerator, logger, save_dir, epoch, data_path,
                         teacher_data_loader, embeddings_dict, ref_size, calibrator='linear', tau=1.0):
    """Evaluate on multiple test groups across multiple seeds"""
    model.eval()
    test_losses = []
    
    with open(os.path.join(data_path, f"data_test.pkl"), "rb") as f:
        test_data = pickle.load(f)
    for test_step, test_group_label in enumerate(test_data):
        all_metrics = []
        all_ref_metrics = []
        
        for seed in [1, 2, 3, 4, 5]:
            log_main_process(accelerator, logger, f"Epoch {epoch}: Evaluating on group {test_group_label} on seed {seed}")
            test_questions, test_rewards, test_group_ids, test_ref_questions, test_ref_values = teacher_data_loader.load_test_data(
                test_data,
                test_group_label,
                ref_size, 
                seed=seed
            )
            
            # Reference set predictions
            if embeddings_dict:
                test_ref_dataset = QuestionEmbeddingDataset(test_group_ids, test_ref_questions, test_ref_values, embeddings_dict)
            else:
                test_ref_dataset = QuestionDataset(test_group_ids, test_ref_questions, test_ref_values)
            
            test_ref_dataloader = DataLoader(
                test_ref_dataset,
                batch_size=args.batch_size_per_gpu,
                shuffle=False,
                drop_last=False
            )
            test_ref_dataloader = accelerator.prepare(test_ref_dataloader)
            save_path = os.path.join(save_dir, "predictions", f"eval_epoch_{epoch+1}_group{test_group_label}_seed{seed}")
            eval_metrics_ref, all_preds_ref, all_targets_ref = evaluate(
                model, tokenizer, test_ref_dataloader,
                test_ref_questions, test_ref_values, accelerator, 
                embeddings_dict=embeddings_dict,
                logger=logger, save_dir=save_path, is_ref=True, tau=tau
            )
            all_ref_metrics.append(eval_metrics_ref)         
                    
            reg = calibrate_predictions(all_preds_ref, all_targets_ref, method=calibrator)

            # Apply transformation to predictions
            
            # Create appropriate dataset based on whether embeddings are available
            if embeddings_dict:
                test_dataset = QuestionEmbeddingDataset(test_group_ids, test_questions, test_rewards, embeddings_dict)
            else:
                test_dataset = QuestionDataset(test_group_ids, test_questions, test_rewards)
            
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=args.batch_size_per_gpu,
                shuffle=False,
                drop_last=False
            )
            
            test_dataloader = accelerator.prepare(test_dataloader)
            
            # Run evaluation
            save_path = os.path.join(save_dir, "predictions", f"eval_epoch_{epoch+1}_group{test_group_label}_seed{seed}")
            eval_metrics_i, _, _ = evaluate(
                model, tokenizer, test_dataloader,
                test_ref_questions, test_ref_values, accelerator, 
                embeddings_dict=embeddings_dict,
                logger=logger, save_dir=save_path, is_ref=False, calibrator=reg, tau=tau
            )
            all_metrics.append(eval_metrics_i)
            
            # Collect test loss for plotting
            if 'mse' in eval_metrics_i:
                test_losses.append(eval_metrics_i['mse'])

        # Log average and variance results across seeds
        if accelerator.is_main_process and wandb.run is not None:
            # record metrics for general predictions
            avg_metrics = {}
            var_metrics = {}
            for metric_name in all_metrics[0].keys():
                values = [m[metric_name] for m in all_metrics]
                mean_val = sum(values) / len(values)
                std_val = (sum((v - mean_val) ** 2 for v in values) / (len(values) - 1)) ** 0.5
                
                if 'interval' in metric_name:
                    wandb.log({
                        f"eval_interval/group_{test_group_label}/avg_{metric_name}": mean_val,
                        f"eval_interval/group_{test_group_label}/std_{metric_name}": std_val,
                        "epoch": epoch + 1
                    })
                else:
                    wandb.log({
                        f"eval/group_{test_group_label}/avg_{metric_name}": mean_val,
                        f"eval/group_{test_group_label}/std_{metric_name}": std_val,
                        "epoch": epoch + 1
                    })
                avg_metrics[metric_name] = mean_val
                var_metrics[metric_name] = std_val
            
            metrics_save_path = os.path.join(save_dir, "metrics", f"eval_epoch_{epoch+1}_group{test_group_label}")
            os.makedirs(metrics_save_path, exist_ok=True)
            with open(os.path.join(metrics_save_path, "inference_results.json"), "w") as f:
                json.dump({"avg_metrics": avg_metrics, "var_metrics": var_metrics}, f, indent=2)
            
            # record metrics for reference predictions
            avg_metrics = {}
            var_metrics = {}
            for metric_name in all_ref_metrics[0].keys():
                values = [m[metric_name] for m in all_ref_metrics]
                mean_val = sum(values) / len(values)
                std_val = (sum((v - mean_val) ** 2 for v in values) / (len(values) - 1)) ** 0.5
                
                if 'interval' in metric_name:
                    wandb.log({
                        f"eval_interval/group_{test_group_label}/avg_{metric_name}": mean_val,
                        f"eval_interval/group_{test_group_label}/std_{metric_name}": std_val,
                        "epoch": epoch + 1
                    })
                else:
                    wandb.log({
                        f"eval/group_{test_group_label}/avg_{metric_name}": mean_val,
                        f"eval/group_{test_group_label}/std_{metric_name}": std_val,
                        "epoch": epoch + 1
                    })
                avg_metrics[metric_name] = mean_val
                var_metrics[metric_name] = std_val
            
            metrics_save_path = os.path.join(save_dir, "metrics", f"eval_epoch_{epoch+1}_group{test_group_label}")
            os.makedirs(metrics_save_path, exist_ok=True)
            with open(os.path.join(metrics_save_path, "inference_results_ref.json"), "w") as f:
                json.dump({"avg_metrics": avg_metrics, "var_metrics": var_metrics}, f, indent=2)

    # Calculate average test loss across all groups and seeds
    if accelerator.is_main_process and test_losses:
        avg_test_loss = sum(test_losses) / len(test_losses)
        # Add to metrics dictionary using a global variable
        global metrics_for_plotting
        metrics_for_plotting["test_losses"].append(avg_test_loss)
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({"avg_test_loss": avg_test_loss, "epoch": epoch + 1})

@torch.no_grad()
def evaluate(model, tokenizer, test_dataloader, ref_questions, ref_values, accelerator, 
             embeddings_dict=None, logger=None, save_dir=None, is_ref=False, calibrator=None, tau=1.0):
    """Evaluate the model on test data"""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    all_preds, all_targets = [], []
    batch_ref_texts = ref_questions
    
    # Pre-process reference texts once
    if embeddings_dict is None:
        ref_enc = tokenizer(
            batch_ref_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        ref_enc = {k: v.to(accelerator.device) for k, v in ref_enc.items()}
    
    # Process test data in batches
    for i, batch_data in enumerate(get_tqdm_iterator(accelerator, test_dataloader, desc="Evaluating")):
        has_embeddings = len(batch_data) == 4
        
        if has_embeddings:  # Using embeddings
            group_ids, batch_questions, batch_targets, batch_embeddings = batch_data
        else:  # Using tokenizer
            group_ids, batch_questions, batch_targets = batch_data
        
        batch_targets = torch.tensor(batch_targets, dtype=torch.float32).to(accelerator.device)
        
        if isinstance(group_ids, torch.Tensor):
            group_ids = group_ids.cpu().tolist()
        
        # Prepare reference values for each group
        batch_ref_values = [
            [ref_values[i] for i in range(len(batch_ref_texts))] for _ in group_ids
        ]
        ref_values_tensor = torch.tensor(batch_ref_values, dtype=torch.float32).to(accelerator.device)
        
        # Forward pass
        if has_embeddings and batch_embeddings[0] is not None:
            batch_embeddings = torch.stack([emb.to(accelerator.device) for emb in batch_embeddings])
            ref_embeddings = torch.stack([embeddings_dict[ref_text].to(accelerator.device) 
                                          for ref_text in batch_ref_texts])
            preds = model(batch_embeddings, ref_embeddings, ref_values_tensor, tau=tau)
        else:
            query_enc = tokenizer(
                batch_questions, 
                padding=True, 
                truncation=True, 
                return_tensors="pt",
            )
            query_enc = {k: v.to(accelerator.device) for k, v in query_enc.items()}
            preds = model(query_enc, ref_enc, ref_values_tensor, tau=tau)
        
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(batch_targets.cpu().tolist())
        
        if i % 10 == 0:
            log_main_process(accelerator, logger, f"Evaluated {i*len(batch_questions)}/{len(test_dataloader.dataset)} samples")

    # Gather results from all processes
    accelerator.wait_for_everyone()
    all_preds_tensor = torch.tensor(all_preds, dtype=torch.float32, device=accelerator.device)
    all_targets_tensor = torch.tensor(all_targets, dtype=torch.float32, device=accelerator.device)
    all_preds_gathered = accelerator.gather_for_metrics(all_preds_tensor).cpu().numpy()
    all_targets_gathered = accelerator.gather_for_metrics(all_targets_tensor).cpu().numpy()

    # Compute metrics
    if accelerator.is_main_process:
        mse = mean_squared_error(all_targets_gathered, all_preds_gathered)
        mae = mean_absolute_error(all_targets_gathered, all_preds_gathered)
        
        try:
            pearson_corr, _ = pearsonr(all_targets_gathered, all_preds_gathered)
            if np.isnan(pearson_corr):
                pearson_corr = 0.0
                log_main_process(accelerator, logger, "Warning: NaN detected in Pearson correlation, setting to 0.0")
        except Exception as e:
            pearson_corr = 0.0
            log_main_process(accelerator, logger, f"Error calculating Pearson correlation: {e}")
        
        try:
            spearman_corr, _ = spearmanr(all_targets_gathered, all_preds_gathered)
            if np.isnan(spearman_corr):
                spearman_corr = 0.0
                log_main_process(accelerator, logger, "Warning: NaN detected in Spearman correlation, setting to 0.0")
        except Exception as e:
            spearman_corr = 0.0
            log_main_process(accelerator, logger, f"Error calculating Spearman correlation: {e}")
        
        metrics = {
            "mse": float(mse),
            "mae": float(mae),
            "pearson": float(pearson_corr),
            "spearman": float(spearman_corr),
        }
        if not is_ref:
            interval_metrics = interval_classification_metrics(all_targets_gathered, all_preds_gathered, B=min(1024,len(all_preds_gathered)//2))
            metrics.update(interval_metrics)
            if calibrator:
                all_preds_transformed = calibrator(all_preds_gathered).reshape(-1)
                mse = mean_squared_error(all_targets_gathered, all_preds_transformed)
                mae = mean_absolute_error(all_targets_gathered, all_preds_transformed)
                pearson_corr, _ = pearsonr(all_targets_gathered, all_preds_transformed)
                spearman_corr, _ = spearmanr(all_targets_gathered, all_preds_transformed)
                metrics.update({
                    "mse_transformed": float(mse),
                    "mae_transformed": float(mae),
                    "pearson_transformed": float(pearson_corr),
                    "spearman_transformed": float(spearman_corr),
                })
                interval_metrics_tranformed = interval_classification_metrics(all_targets_gathered, all_preds_transformed, B=min(1024,len(all_preds_gathered)//2))
                for key, value in interval_metrics_tranformed.items():
                    metrics[f"{key}_transformed"] = value
        # Log metrics
        log_msg = f"\nTest MSE: {mse:.4f}, MAE: {mae:.4f}, Pearson: {pearson_corr:.4f}, Spearman: {spearman_corr:.4f}"
        accelerator.print(log_msg)
        log_main_process(accelerator, logger, log_msg)
        
        # Save predictions and metrics
        if save_dir:
            # Only save numpy arrays if needed (they can be large)
            if args.save_predictions:
                if is_ref:
                    pred_path = os.path.join(save_dir, "predictions_ref.npy")
                    target_path = os.path.join(save_dir, "targets_ref.npy")
                else:
                    pred_path = os.path.join(save_dir, "predictions.npy")
                    target_path = os.path.join(save_dir, "targets.npy")
                np.save(pred_path, all_preds_gathered)
                np.save(target_path, all_targets_gathered)
            
            log_main_process(accelerator, logger, f"Inference results saved to {save_dir}")
        
        return metrics, all_preds_gathered, all_targets_gathered
    
    return {}, [], []

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
if __name__ == "__main__":
    # Add argument parsing for model configuration
    parser = argparse.ArgumentParser(description="Train a few-shot model for regression")
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "binary_cross_entropy", "pinball","focal_binary_cross_entropy"],
                        help="Loss function to use")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="Transformer model to use")
    parser.add_argument("--use_embeddings", action="store_true", help="Use pre-computed embeddings")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--batch_size_per_gpu", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--ref_size", type=int, default=128, help="Reference size for few-shot learning")
    parser.add_argument("--data_path", type=str, default="./datasets", help="Data path")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze the encoder parameters")
    parser.add_argument("--left_padding", action="store_true", help="Use left padding for tokenization")
    parser.add_argument("--method", type=str, default='residual', help="Method to use")
    parser.add_argument("--scaling", type=str, default='platt', choices=['platt', 'temperature', 'group_logit_temp','plain'], help="Scaling method")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers in regression head")
    parser.add_argument("--hidden_size", type=int, default=896, help="Hidden size for regression head")
    parser.add_argument("--save_predictions", action="store_true", help="Save prediction arrays")
    parser.add_argument("--use_scheduler", action="store_true", help="Use learning rate scheduler")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps for scheduler")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=1, help="Warmup steps for scheduler")
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--top_k", type=int, default=None, help="Top k for few-shot learning")
    parser.add_argument("--tau", type=float, default=1.0, help="Tau for few-shot learning")
    parser.add_argument("--use_layernorm", action="store_true", help="placeholder")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--calibrator", type=str, default='linear', choices=['linear', 'isotonic', 'platt'], help="Calibrator for few-shot learning")
    args = parser.parse_args()

    set_seed(args.seed)
    
    # Initialize accelerator for distributed training
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    log_main_process(accelerator, None, f"Number of processes: {accelerator.num_processes}")
    
    # Set up paths and configuration
    data_path = args.data_path
    save_path = f"./{args.output_dir}/stage2_results_{args.loss_type}"
    model_name = args.model_name
    learning_rate = args.lr
    batch_size = args.batch_size_per_gpu * accelerator.num_processes
    loss_type = args.loss_type
    ref_size = args.ref_size
    hidden_size = args.hidden_size
    has_embeddings = args.use_embeddings
       

    # Create output directory
    os.makedirs(save_path, exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(save_path, "training.log")),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("few_shot_model")

    # Save experiment configuration
    config = {
        "model_name": model_name,
        "ref_size": ref_size,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "batch_size_per_gpu": args.batch_size_per_gpu,
        "loss_type": loss_type,
        'data_path': data_path,
        'has_embeddings': has_embeddings,
        'freeze_encoder': args.freeze_encoder,
        'method': args.method,
        'num_layers': args.num_layers,
        'use_scheduler': args.use_scheduler,
        'warmup_steps': args.warmup_steps if args.use_scheduler else 0,
        'scaling': args.scaling,
        'tau': args.tau,
        'hidden_size': hidden_size,
        'calibrator': args.calibrator
    }
    
    log_main_process(accelerator, logger, f"Config: {config}")

    # Create specific directory for this experiment
    experiment_name = (
        f"model_{model_name.split('/')[-1].replace('-', '_')}"
        f"_ref{ref_size}_batch{batch_size}_lr{learning_rate}"
        f"_freeze{args.freeze_encoder}_method{args.method}_layers{args.num_layers}_scaling{args.scaling}_tau{args.tau}_calibrator{args.calibrator}"
    )
    save_path = os.path.join(save_path, experiment_name)
    os.makedirs(save_path, exist_ok=True)

    # Initialize wandb for experiment tracking
    if accelerator.is_main_process:
        pass
        # wandb_run_name = f"test"
        
        # wandb.init(
        #     project="test",
        #     name=wandb_run_name,
        #     config=config
        # )
        
        # log_main_process(accelerator, logger, f"Initialized wandb run: {wandb.run.name}")

    # Save config for reproducibility
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # Initialize model, tokenizer and optimizer
    log_main_process(accelerator, logger, "Initializing model and loading data...")
    
    if args.loss_type == 'pinball':
        raise NotImplementedError("Pinball loss is not implemented")
    else:
        model = FewShotRegressor(
            model_name, 
            method=args.method, 
            num_layers=args.num_layers, 
            has_embeddings=has_embeddings,
            lora=args.lora,
            scaling=args.scaling,
            top_k=args.top_k,
            hidden_size=hidden_size
        )

    # Freeze encoder if specified
    if args.freeze_encoder:
        log_main_process(accelerator, logger, "Freezing encoder parameters")
        if hasattr(model, 'encoder'):
            for param in model.encoder.encoder.parameters():
                param.requires_grad = False
    
    # Log trainable parameters
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_main_process(accelerator, logger, f"Trainable parameters: {num_trainable:,}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if args.left_padding:
        tokenizer.padding_side = "left"
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
    )
    
    # Load training data
    teacher_data_loader = TeacherDataset(data_path,model_name)
    train_questions, train_rewards, train_group_ids, ref_candidate_questions, ref_candidate_labels = teacher_data_loader.load_train_data()

    # Create appropriate dataset and dataloader
    embeddings_dict = None
    if has_embeddings:
        embeddings_dict = teacher_data_loader.load_embeddings()
        log_main_process(accelerator, logger, f"Loaded embeddings for {len(embeddings_dict)} questions")
        train_dataset = QuestionEmbeddingDataset(train_group_ids, train_questions, train_rewards, embeddings_dict)
    else:
        train_dataset = QuestionDataset(train_group_ids, train_questions, train_rewards)

    # Initialize dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=True,
        drop_last=False,
    )

    # Setup learning rate scheduler if requested
    scheduler = None
    if args.use_scheduler:
        # Calculate total training steps
        total_steps = len(train_dataloader) * args.epochs
        warmup_steps = min(args.warmup_steps, total_steps // 10)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        log_main_process(accelerator, logger, 
                        f"Using scheduler with {warmup_steps} warmup steps out of {total_steps} total steps")
    
    # Prepare for distributed training
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )
    if scheduler:
        scheduler = accelerator.prepare(scheduler)

    # Start training
    log_main_process(accelerator, logger, f"Starting training with loss_type: {loss_type}")
    train_metrics = train_full_epoch(
        model=model,
        train_dataloader=train_dataloader,
        ref_questions=ref_candidate_questions,
        ref_values=ref_candidate_labels,
        optimizer=optimizer,
        accelerator=accelerator,
        embeddings_dict=embeddings_dict,
        teacher_data_loader=teacher_data_loader,
        epochs=args.epochs,
        logger=logger,
        save_dir=save_path,
        loss_type=args.loss_type,
        ref_size=args.ref_size,
        scheduler=scheduler,
        tau=args.tau,
        calibrator=args.calibrator,
    )
    
    # Plot and save training loss curve
    if accelerator.is_main_process:
        plt.figure(figsize=(10, 6))
        plt.plot(train_metrics["epoch_losses"], label='Training Loss')
        
        # Add test loss curve if available
        if "test_losses" in train_metrics and train_metrics["test_losses"]:
            # Create x-axis points for test losses (every 2 epochs)
            test_epochs = list(range(0, args.epochs, 2))
            if len(test_epochs) > len(train_metrics["test_losses"]):
                test_epochs = test_epochs[:len(train_metrics["test_losses"])]
            plt.plot(test_epochs, train_metrics["test_losses"], label='Test Loss', linestyle='--', marker='o')
        
        plt.title(f"Training and Test Loss per Epoch ({loss_type})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_path, "loss_curves.png"))
        logger.info(f"Loss curves plot saved to {os.path.join(save_path, 'loss_curves.png')}")
        
        # Log the plot to wandb
        if wandb.run is not None:
            wandb.log({"loss_curves_plot": wandb.Image(plt)})
            plt.close()
            
        # Close wandb
        wandb.finish()
        logger.info("Wandb run finished")