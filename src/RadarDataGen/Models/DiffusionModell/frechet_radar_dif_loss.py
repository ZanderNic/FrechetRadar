# std lib imports
from typing import Tuple, Optional, Dict

# 3 party import
import torch

# projekt imports



def weighted_mse_loss(
    real: torch.Tensor,
    pred: torch.Tensor,
    batch_weights: Optional[torch.Tensor] = None,
    channel_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
        Compute a weighted Mean Squared Error (MSE) loss for epsilon prediction in a diffusion model.
        
        This function applies separate weights to batches and channels. The common use case is to give
        the first channel (e.g., a presence/validity indicator) a higher importance than the other feature
        channels. It returns both the total weighted loss and per-component metrics for logging.

        Args:
            real (torch.Tensor):
                Ground truth tensor of shape (B, C, H, W).
            pred (torch.Tensor):
                Predicted tensor of the same shape (B, C, H, W).
            batch_weights (torch.Tensor, optional):
                1D tensor of shape (B,) with per-sample weights. If None, defaults to ones.
            channel_weights (torch.Tensor, optional):
                1D tensor of shape (C,) with per-channel weights. If None, defaults to ones.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]:
                - total_loss (torch.Tensor): Scalar weighted MSE loss (presence + features).
                - metrics_dict (dict): 
                    {
                        "Presence_Loss": float,
                        "Feature_Loss": float
                    }
    """
    if batch_weights is None:
        batch_weights = torch.ones(pred.shape[0], device=pred.device)

    if channel_weights is None:
        channel_weights = torch.ones(pred.shape[1], device=pred.device)

    batch_weights = batch_weights.view(pred.shape[0], 1, 1, 1).to(pred.device)              # Shape: B, 1, 1, 1
    channel_weights = channel_weights.view(1, pred.shape[1], 1, 1).to(pred.device)          # Shape: 1, C, 1, 1

    mse = ((pred - real) ** 2) * batch_weights * channel_weights

    valid_dim_loss = (mse[:, 0, :, :]).sum() / torch.numel(mse) 
    feature_loss = (mse[:, 1:, :, :]).sum() / torch.numel(mse)

    return valid_dim_loss + feature_loss, {"Valid_Dim_Loss": valid_dim_loss.item(), "Feature_Loss": feature_loss.item()}



def mixed_ce_mse_loss_x0_pred(
    real: torch.Tensor,
    pred: torch.Tensor,
    batch_weights: Optional[torch.Tensor] = None,
    channel_weights: Optional[torch.Tensor] = None,
    mse_weight: float = 1,
    ce_weight: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
        Compute a combined loss for x₀ predictions:
        - Binary Cross-Entropy (with logits) for channel 0 (binary classification: point presence)
        - Mean Squared Error (MSE) for all remaining channels (feature values)

        Args:
            real (torch.Tensor): 
                Ground truth tensor of shape (B, C, H, W)
            pred (torch.Tensor): 
                Predicted tensor of the same shape  (B, C, H, W)
            batch_weights (Optional[torch.Tensor]): 
                Optional weights per batch element (shape: B)
            channel_weights (Optional[torch.Tensor]): 
                Optional weights per channel (shape: C)

        Returns:
            tuple:
                - torch.Tensor: Scalar combined loss value (CE + MSE)
                - dict: Dictionary with individual loss components:
                    {
                        "Valid_Dim_Loss": Cross-Entropy loss for channel 0,
                        "Feature_Loss": MSE loss for channels 1..C-1
                    }
    """

    if batch_weights is None:
        batch_weights = torch.ones(pred.shape[0], device=pred.device)

    if channel_weights is None:
        channel_weights = torch.ones(pred.shape[1], device=pred.device)

    batch_weights = batch_weights.view(real.shape[0], 1, 1, 1).to(pred.device)              # Shape: B, 1, 1, 1
    channel_weights = channel_weights.view(1, pred.shape[1], 1, 1).to(pred.device)          # Shape: 1, C, 1, 1
    
    pred_clamped = pred[:, 0, :, :].clamp(1e-5, 1)     # becasue pred can be any real number we need to clamp it for cross entropy loss 

    cros_entropy_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_clamped , real[:, 0, :, :].float(), reduction='none')         # for the first dim we want cross entropy loss this will be shape # B, H, W
    valid_dim_loss = (cros_entropy_loss.unsqueeze(1) * batch_weights *  channel_weights[:, 0, :, :]).mean() * ce_weight                         # add weights to cross entropy loss
    feature_loss = (((pred[:, 1:, :, :] - real[:, 1:, :, :]) ** 2) * batch_weights * channel_weights[:, 1:, :, :]).mean() * mse_weight          # for the rest we take mse loss
    
    return valid_dim_loss + feature_loss, {"Valid_Dim_Loss":  valid_dim_loss, "Feature_Loss": feature_loss}



def presence_aware_weighted_mse_loss(
    real: torch.Tensor,
    pred: torch.Tensor,
    batch_weights: Optional[torch.Tensor] = None,
    channel_weights: Optional[torch.Tensor] = None,
    valid_indc: float = 1.0,
    valid_mse_weight: float = 0.95,
    non_valid_mse_weight: float = 0.05,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
        Compute a presence-aware, weighted Mean Squared Error (MSE) loss for epsilon prediction in a diffusion model.

        This loss:
        - applies per-batch and per-channel weights,
        - treats channel 0 as a presence/validity indicator,
        - weights feature-channel errors differently for valid vs. non-valid pixels using a binary mask
            derived from the presence channel in the ground truth.

        Args:
            real (torch.Tensor): 
                Ground-truth tensor of shape (B, C, H, W).
            pred (torch.Tensor): 
                Predicted tensor of shape (B, C, H, W).
            batch_weights (torch.Tensor, optional): 
                1D tensor of shape (B,) with per-sample weights. If None, defaults to ones.
            channel_weights (torch.Tensor, optional): 
                1D tensor of shape (C,) with per-channel weights. If None, defaults to ones.
            valid_indc (float): 
                Threshold/value used to mark presence/validity in channel 0 of `real`. Currently using equality: real[:, 0, :, :] == valid_indc.)
            valid_mse_weight (float): 
                Weight applied to feature-channel MSE on pixels marked as valid.
            non_valid_mse_weight (float): 
                Weight applied to feature-channel MSE on pixels marked as non-valid.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]:
                - total_loss (torch.Tensor): Scalar weighted MSE loss (presence + features).
                - metrics_dict (dict):
                    {
                        "Presence_Loss": float,
                        "Feature_Loss": float
                    }
    """
    if batch_weights is None:
        batch_weights = torch.ones(pred.shape[0], device=pred.device)

    if channel_weights is None:
        channel_weights = torch.ones(pred.shape[1], device=pred.device)

    batch_weights = batch_weights.view(pred.shape[0], 1, 1, 1).to(pred.device)                  # Shape: (B, 1, 1, 1)
    channel_weights = channel_weights.view(1, pred.shape[1], 1, 1).to(pred.device)              # Shape: (1, C, 1, W)

    valid_mask = (real[:, 0:1, :, :] == valid_indc).float()                                     # Shape: (B, 1, H, W)
    inverted_mask = 1.0 - valid_mask                                                            # Shape: (B, 1, H, W)
    feature_weights = valid_mask * valid_mse_weight + inverted_mask * non_valid_mse_weight      # Shape: (B, 1, H, W)

    mse = ((pred - real) ** 2) * batch_weights * channel_weights                                # Shape: (B, C, H, W)

    valid_dim_loss = (mse[:, 0, :, :]).sum() / torch.numel(mse) 
    feature_loss = (mse[:, 1:, :, :] * feature_weights).sum() / torch.numel(mse)

    return valid_dim_loss + feature_loss, {"Valid_Dim_Loss": valid_dim_loss.item(), "Feature_Loss": feature_loss.item()}



def presence_aware_ce_mse_loss_x0_pred(
    real: torch.Tensor,
    pred: torch.Tensor,
    batch_weights: Optional[torch.Tensor] = None,
    channel_weights: Optional[torch.Tensor] = None,
    valid_mse_weight: float = 0.90,
    non_valid_mse_weight: float = 0.10,
    mse_weight: float = 1,
    ce_weight: float = 0.2,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
        Compute a presence-aware combined loss for x₀ predictions.

        This loss combines:
        1) Binary Cross-Entropy with logits (BCE) on channel 0 to model presence/validity.
        2) Mean Squared Error (MSE) on channels 1..C-1 (feature channels), where the per-pixel
            contribution is weighted differently for valid vs. non-valid pixels according to a
            binary mask derived from the ground-truth presence channel.

        The function additionally supports per-sample (batch) and per-channel weighting, and
        global scaling of the BCE/MSE contributions via `ce_weight` and `mse_weight`.

        Args:
            real (torch.Tensor):
                Ground-truth tensor of shape (B, C, H, W).
                Channel 0 is the presence/validity target (expected to be 0/1 or match `valid_indc`).
            pred (torch.Tensor):
                Prediction tensor of shape (B, C, H, W).
                Channel 0 is treated as logits for BCE (i.e., passed to `binary_cross_entropy_with_logits`).
            batch_weights (Optional[torch.Tensor], default=None):
                1D tensor of shape (B,) with per-sample weights. If None, uses ones.
            channel_weights (Optional[torch.Tensor], default=None):
                1D tensor of shape (C,) with per-channel weights. If None, uses ones.
            valid_mse_weight (float, default=0.10):
                Weight applied to the feature-channel MSE at pixels marked as valid by the mask.
            non_valid_mse_weight (float, default=0.90):
                Weight applied to the feature-channel MSE at pixels marked as non-valid by the mask.
            mse_weight (float, default=1):
                Global scalar multiplier applied to the feature MSE term.
            ce_weight (float, default=1):
                Global scalar multiplier applied to the presence BCE term.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]:
                - total_loss (torch.Tensor): Scalar loss = `ce_weight * BCE_presence + mse_weight * MSE_features`.
                - metrics (Dict[str, float]): Individual components for logging:
                    {
                        "Valid_Dim_Loss": float,   # BCE (presence) term after weighting and reduction
                        "Feature_Loss": float      # MSE (features) term after presence-aware weighting and reduction
                    }
    """
    valid_indc = 1.0    # this is the only setting that makes sense for crosentropy setting

    if batch_weights is None:
        batch_weights = torch.ones(pred.shape[0], device=pred.device)

    if channel_weights is None:
        channel_weights = torch.ones(pred.shape[1], device=pred.device)

    batch_weights = batch_weights.view(real.shape[0], 1, 1, 1)
    channel_weights = channel_weights.view(1, pred.shape[1], 1, 1)
    
    valid_mask = (real[:, 0:1, :, :] == valid_indc).float()                                     # Shape: (B, 1, H, W)
    inverted_mask = 1.0 - valid_mask                                                            # Shape: (B, 1, H, W)
    feature_weights = valid_mask * valid_mse_weight + inverted_mask * non_valid_mse_weight      # Shape: (B, 1, H, W)

    pred_clamped = pred[:, 0, :, :].clamp(1e-5, 1)     # becasue pred can be any real number we need to clamp it for cross entropy loss 

    cros_entropy_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_clamped , real[:, 0, :, :].float(), reduction='none')                         # for the first dim we want cross entropy loss this will be shape # B, H, W
    valid_dim_loss = (cros_entropy_loss.unsqueeze(1) * batch_weights *  channel_weights[:, 0, :, :]).mean()  * ce_weight                                        # add weights to cross entropy loss
    feature_loss = ((((pred[:, 1:, :, :] - real[:, 1:, :, :]) ** 2) * batch_weights * channel_weights[:, 1:, :, :]) * feature_weights).mean() * mse_weight      # for the rest we take mse loss
    
    return valid_dim_loss + feature_loss, {"Valid_Dim_Loss":  valid_dim_loss, "Feature_Loss": feature_loss}