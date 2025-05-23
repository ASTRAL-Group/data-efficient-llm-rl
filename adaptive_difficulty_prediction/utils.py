import torch
from tqdm import tqdm
import os
import pickle
from torch.utils.data import Dataset


def interval_classification_metrics(targets, preds, B=1024):
    import numpy as np
    from sklearn.metrics import precision_score, recall_score
    results = {}
    
    # centers = [0.25, 0.5, 0.75, 0.4, 0.6]
    # closed interval
    interval_mapping = [
        (0.5, 0.125, 0.875),
    ]
    
    # Center-based metrics
    for center, interval_lower, interval_upper in interval_mapping:
        
        pred_dist = np.abs(preds - center)
        pred_labels = np.zeros_like(preds, dtype=int)
        topB_indices = np.argsort(pred_dist)[:B]
        pred_labels[topB_indices] = 1
        
        in_interval = (targets >= interval_lower) & (targets <= interval_upper)
        target_labels = in_interval.astype(int)
        ub_interval = min(np.sum(target_labels)/B, 1)
        
        precision = precision_score(target_labels, pred_labels, zero_division=0)
        results[f"precision_center_{center}_interval_{interval_lower}_{interval_upper}"] = round(float(precision), 4)
        results[f"precision_gap_center_{center}_interval_{interval_lower}_{interval_upper}"] = ub_interval - round(float(precision), 4)

    # Random baseline per interval
    all_intervals = set([(interval_lower, interval_upper) for _, interval_lower, interval_upper in interval_mapping])
    for interval_lower, interval_upper in all_intervals:
        in_interval = (targets >= interval_lower) & (targets <= interval_upper)
        target_labels = in_interval.astype(int)
        ub_interval = min(np.sum(target_labels)/B, 1)
        
        pred_random_labels = np.zeros_like(preds, dtype=int)
        random_indices = np.random.choice(len(preds), size=B, replace=False)
        pred_random_labels[random_indices] = 1
        
        random_precision = precision_score(target_labels, pred_random_labels, zero_division=0)
        
        results[f"ub_interval_{interval_lower}_{interval_upper}"] = ub_interval
        results[f"precision_random_interval_{interval_lower}_{interval_upper}"] = round(float(random_precision), 4)

    # MAE centers
    mse_centers = [0.25, 0.5, 0.75]
    for center in mse_centers:
        pred_dist = np.abs(preds - center)
        topB_indices = np.argsort(pred_dist)[:B]
        absolute_error = np.mean(np.abs(targets[topB_indices] - center))
        results[f"mae_center_{center}"] = round(float(absolute_error), 4)
        
    return results

def interval_prediciton_evaluate(targets, preds, B=1024):
    import numpy as np
    from sklearn.metrics import precision_score, recall_score
    results = {}
    
    # closed interval
    interval_mapping = [
        (0.5, 0.125, 0.875),
    ]
    
    preds_lb = preds[:, 0]
    preds_ub = preds[:, 1]
    interval_centers = (preds_lb + preds_ub) / 2
    
    for center, interval_lower, interval_upper in interval_mapping:
        # Step 1: Fully contained intervals
        fully_contained_mask = (preds_lb >= interval_lower) & (preds_ub <= interval_upper)
        fully_contained_indices = np.where(fully_contained_mask)[0]
        print(f"fully_contained_indices: {len(fully_contained_indices)}")

        # Step 2: Sort them by center proximity
        contained_dists = np.abs(interval_centers[fully_contained_indices] - center)
        sorted_contained_indices = fully_contained_indices[np.argsort(contained_dists)]

        if len(sorted_contained_indices) >= B:
            selected_indices = sorted_contained_indices[:B]
        else:
            num_needed = B - len(sorted_contained_indices)

            # Step 3: Remaining candidates
            all_indices = np.arange(len(preds))
            remaining_indices = np.setdiff1d(all_indices, sorted_contained_indices)

            # Filtering priority: closer to interval constraint
            mask = np.ones_like(remaining_indices, dtype=bool)

            if interval_lower == 0.125:
                mask &= preds_lb[remaining_indices] > 0.125
            if interval_upper == 0.875:
                mask &= preds_ub[remaining_indices] < 0.875

            preferred_indices = remaining_indices[mask]
            fallback_indices = remaining_indices[~mask]

            # Sort by center distance
            preferred_dists = np.abs(interval_centers[preferred_indices] - center)
            preferred_sorted = preferred_indices[np.argsort(preferred_dists)]

            fallback_dists = np.abs(interval_centers[fallback_indices] - center)
            fallback_sorted = fallback_indices[np.argsort(fallback_dists)]

            # Combine preferred first, then fallback
            combined_sorted = np.concatenate([preferred_sorted, fallback_sorted])
            supplement_indices = combined_sorted[:num_needed]

            # Final selected
            selected_indices = np.concatenate([sorted_contained_indices, supplement_indices])

        # Label predicted selections
        pred_labels = np.zeros(len(preds), dtype=int)
        pred_labels[selected_indices] = 1

        # Label ground truth
        target_mask = (targets >= interval_lower) & (targets <= interval_upper)
        target_labels = target_mask.astype(int)
        ub_interval = min(np.sum(target_labels) / B, 1)

        precision = precision_score(target_labels, pred_labels, zero_division=0)
        results[f"precision_center_{center}_interval_{interval_lower}_{interval_upper}"] = round(float(precision), 4)
        results[f"precision_gap_center_{center}_interval_{interval_lower}_{interval_upper}"] = ub_interval - round(float(precision), 4)

        pred_labels = np.zeros(len(preds), dtype=int)
        pred_labels[selected_indices] = 1

        in_interval = (targets >= interval_lower) & (targets <= interval_upper)
        target_labels = in_interval.astype(int)
        ub_interval = min(np.sum(target_labels)/B, 1)

        precision = precision_score(target_labels, pred_labels, zero_division=0)
        results[f"precision_center_{center}_interval_{interval_lower}_{interval_upper}"] = round(float(precision), 4)
        results[f"precision_gap_center_{center}_interval_{interval_lower}_{interval_upper}"] = ub_interval - round(float(precision), 4)

    # Random baseline per interval
    all_intervals = set([(interval_lower, interval_upper) for _, interval_lower, interval_upper in interval_mapping])
    for interval_lower, interval_upper in all_intervals:
        in_interval = (targets >= interval_lower) & (targets <= interval_upper)
        target_labels = in_interval.astype(int)
        ub_interval = min(np.sum(target_labels)/B, 1)
        
        pred_random_labels = np.zeros_like(preds[:,0], dtype=int)
        random_indices = np.random.choice(len(preds), size=B, replace=False)
        pred_random_labels[random_indices] = 1
        random_precision = precision_score(target_labels, pred_random_labels, zero_division=0)
        
        results[f"ub_interval_{interval_lower}_{interval_upper}"] = ub_interval
        results[f"precision_random_interval_{interval_lower}_{interval_upper}"] = round(float(random_precision), 4)

    # MAE centers
    mse_centers = [0.25, 0.5, 0.75]
    preds_lb = preds[:, 0]
    preds_ub = preds[:, 1]
    pred_centers = (preds_lb + preds_ub) / 2

    for center in mse_centers:
        pred_dist = np.abs(pred_centers - center)
        topB_indices = np.argsort(pred_dist)[:B]
        absolute_error = np.mean(np.abs(targets[topB_indices] - center))
        results[f"mae_center_{center}"] = round(float(absolute_error), 4)
        
    return results
    

def log_main_process(accelerator, logger, message, level="info"):
    """Log only on the main process to avoid duplicate logs"""
    if accelerator.is_main_process:
        if logger is None:
            # Just print to console if no logger is provided
            print(message)
            return
            
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "debug":
            logger.debug(message)
            
# Update tqdm to only show on main process
def get_tqdm_iterator(accelerator, iterable, **kwargs):
    """Returns a tqdm iterator that only displays on the main process"""
    if accelerator.is_main_process:
        return tqdm(iterable, **kwargs)
    else:
        return iterable

import torch

def calibrate_predictions(preds, targets, method='linear'):
    """
    Calibrate predicted probabilities using a specified method.

    Args:
        preds (np.ndarray): shape [N], predicted probabilities (must be in (0,1))
        targets (np.ndarray): shape [N], soft or hard labels in [0,1]
        method (str): one of 'linear', 'isotonic', or 'platt'

    Returns:
        calibrated_fn: function that maps new predicted probabilities to calibrated ones
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from scipy.optimize import minimize
    import numpy as np
    preds = preds.reshape(-1)
    targets = targets.reshape(-1)

    if method == 'linear':
        X = preds.reshape(-1, 1)
        y = targets
        reg = LinearRegression()
        reg.fit(X, y)
        slope = reg.coef_[0]
        intercept = reg.intercept_

        def calibrated_fn(x):
            logit = slope * x + intercept
            return logit

        print(f"[Linear] y = {slope:.4f} * x + {intercept:.4f}")

    elif method == 'isotonic':
        # pad to enforce boundary behavior
        X = np.concatenate([[0.0], preds, [1.0]])
        y = np.concatenate([[0.0], targets, [1.0]])
        reg = IsotonicRegression(out_of_bounds='clip')
        reg.fit(X, y)

        def calibrated_fn(x):
            return reg.predict(x)

        print("[Isotonic] fitted isotonic regression")

    elif method == 'platt':
        # Clip only for logit computation
        eps = 1e-6
        preds_clip = np.clip(preds, eps, 1 - eps)
        targets_clip = np.clip(targets, eps, 1 - eps)
        logits = np.log(preds_clip / (1 - preds_clip))

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        def loss_fn(params):
            a, b = params
            z = a * logits + b
            prob = sigmoid(z)
            return -np.mean(targets_clip * np.log(prob + eps) +
                            (1 - targets_clip) * np.log(1 - prob + eps))

        res = minimize(loss_fn, x0=[1.0, 0.0], method='L-BFGS-B')
        a_opt, b_opt = res.x

        def calibrated_fn(x):
            x = np.clip(np.asarray(x), eps, 1 - eps)
            logit = np.log(x / (1 - x))
            return 1 / (1 + np.exp(-(a_opt * logit + b_opt)))

        print(f"[Platt] soft-label sigmoid: a = {a_opt:.4f}, b = {b_opt:.4f}")

    else:
        raise ValueError(f"Invalid method: {method}")

    return calibrated_fn