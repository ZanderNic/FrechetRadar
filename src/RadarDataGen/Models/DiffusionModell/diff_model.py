# std lib imports
import math
from typing import Callable, Optional, Tuple, List
import os 
import copy

# 3 party import
import torch
from tqdm import tqdm
import pandas as pd 
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from fvcore.nn import FlopCountAnalysis, flop_count_table

# projekt imports
from RadarDataGen.Models.DiffusionModell.schedules import NoiseSchedule
from RadarDataGen.Models.DiffusionModell.frechet_radar_dif_loss import weighted_mse_loss

# GPU-SPEED SETTINGS
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

class DiffusionModel(torch.nn.Module):
    """
        Initializes the DiffusionModel.

        Parameters
            function_approximator (torch.nn.Module):
                Neural network used to predict noise or other targets during training.
            output_data_shape (tuple):
                Shape of the data to be generated, e.g. (3, 64, 64) for RGB images.
            time_steps (int):
                Number of diffusion steps used in the forward and reverse process.
            sinus_time_embeding_dim (int):
                Dimensionality of the sinusoidal time embedding. Should be > 2 else there will be an error 
            mlp_time_embeding_dim (int):
                Dimensionality of the MLP output used to embed time information.
            schedule_type (str):
                Type of noise schedule to use ("linear", "cosine", "sigmoid", etc.).
            device (str):
                Device to run the model on ("cuda" or "cpu").
    """

    def __init__(
        self, 
        function_approximator: torch.nn.Module,
        prediction_type: str = "eps",                    # "x0" or "eps"
        output_data_shape: tuple = (3, 64, 64),
        time_steps: int = 1_000,
        sinus_time_embeding_dim: int = 256, 
        mlp_time_embeding_dim: int = 1024,
        schedule_type: str = "cosine",
        use_ema: bool = True,
        ema_decay: float = 0.999,
        loss_func: Callable = None,
        compile_model: bool = True,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.device = device

        self.model = function_approximator.to(device)
        self.output_data_shape = output_data_shape                  # the function aproximator should take this as input 

        # Prediction type eather "x0" or "eps"
        if prediction_type not in  ["x0", "eps"]:
            raise ValueError(f"prediction type needs to be eather 'x0' or 'eps' but is {prediction_type}")
        self.prediction_type = prediction_type

        # Time steps time and time embeding 
        self.time_steps = time_steps
        self.sinus_time_embeding_dim = sinus_time_embeding_dim
        self.time_embeding_dim = mlp_time_embeding_dim
        self.time_embed_mlp = torch.nn.Sequential(
                    torch.nn.Linear(self.sinus_time_embeding_dim, int(mlp_time_embeding_dim / 2), device=device),
                    torch.nn.SiLU(),
                    torch.nn.Linear(int(mlp_time_embeding_dim / 2), mlp_time_embeding_dim, device=device),
                )
        self.half_dim = self.sinus_time_embeding_dim // 2
        self.freq = torch.exp(torch.arange(self.half_dim, dtype=torch.float32) * -(math.log(10_00.0)) / (self.half_dim -1)).to(self.device)

        # noise scedule 
        self.scedule_type = schedule_type
        scedule = NoiseSchedule(time_steps=time_steps, schedule_type=schedule_type)
        self.betas = scedule.betas.to(self.device)
        self.alphas = scedule.alphas.to(self.device)
        self.acc_alphas = scedule.acc_alphas.to(self.device)
        
        # trainable params
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad] + [p for p in self.time_embed_mlp.parameters() if p.requires_grad]

        # ema 
        self.use_ema = use_ema
        if use_ema:
            self.ema_decay = ema_decay            
            self.ema_model = copy.deepcopy(self.model)
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

            self.ema_embed = copy.deepcopy(self.time_embed_mlp)
            for p in self.ema_embed.parameters():
                p.requires_grad_(False)
            
            if compile_model:
                self.comp_ema_model = torch.compile(self.ema_model, mode="reduce-overhead")
                self.comp_ema_time  = torch.compile(self.ema_embed, mode="reduce-overhead")
            else:
                self.comp_ema_model= None
                self.comp_ema_time  = None

        else:
            self.ema_model = None
            self.ema_embed = None
            self.comp_ema_model= None
            self.comp_ema_time  = None

        if compile_model:
            self.comp_model = torch.compile(self.model, mode="reduce-overhead")
            self.comp_time  = torch.compile(self.time_embed_mlp, mode="reduce-overhead")

        else:
            self.comp_model = self.comp_time = None
            
        # loss func
        self.loss_func = weighted_mse_loss if loss_func is None else loss_func

        # check what kind of loss function we have a normal loss function that returnes just the loss or a loss function that returns [loss, {loss_1: ..., loss_2}] where the dict is only for logging 
        self.loss_func = self.wrap_loss_function()

    
    def wrap_loss_function(self) -> Callable:
        """
            Wrap the provided loss function so that it always returns (loss, metrics).
            If the original loss function returns only loss, wrap it to return (loss, None).
            If it returns more than two values, raise an error.
        """
        original_loss_fn = self.loss_func

        # Test with dummy data
        dummy_pred = torch.randn(2, *self.output_data_shape, device=self.device)
        dummy_target = torch.randn(2, *self.output_data_shape, device=self.device)

        try:
            result = original_loss_fn(dummy_target, dummy_pred)
        except Exception as e:
            raise ValueError(f"Loss function failed during test call: {e}")

        if isinstance(result, tuple):                                                               # Decide wrapping based on result type
            if len(result) == 2 and isinstance(result[1], dict):
                return original_loss_fn                                                             # Already returns (loss, metrics)
            else:
                raise ValueError("Loss function returns an unsupported number of values.")
        else:
            def wrapped_loss(real, pred):                                                        # Wrap to return (loss, None)
                return original_loss_fn(real, pred), {}
            return wrapped_loss


    def raw_time_embeding(self, time_steps: torch.Tensor):
        emb = time_steps.type(torch.float32)[:, None] * self.freq[None, :]
        emb = torch.concat([torch.sin(emb), torch.cos(emb)], dim = 1)
        return emb
    

    @torch.no_grad()
    def _sample_ddpm(
        self,
        num_samples: int,
        use_ema_model: bool = False
    ):
        """
            Generates samples from the learned reverse diffusion process.

            Parameters
                num_smaples (int):
                    Number of samples to generate.

            Returns
                torch.Tensor:
                    Generated samples of shape (num_samples, *output_data_shape).

            Source: 
                Algorithm 2 with EQ (11) from https://arxiv.org/abs/2006.11239 
        """
        model, time_embed_mlp = self._get_active_model_eval(use_ema_model= use_ema_model)
        
        x_t = torch.randn((num_samples, *self.output_data_shape), device=self.device)

        for t in range(self.time_steps, 0, -1):
            z = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)
            
            beta_t = self.betas[t - 1].reshape(-1,1,1,1)
            alpha_t = self.alphas[t - 1].reshape(-1,1,1,1)
            acc_alpha_t = self.acc_alphas[t - 1].reshape(-1,1,1,1)

            emb = self.raw_time_embeding(torch.full((num_samples,), t, device=self.device, dtype=torch.long))
            time_embedding = time_embed_mlp(emb) 

            model_out = model(x_t, time_embedding)
            if self.prediction_type == "eps":
                eps_hat = model_out
            elif self.prediction_type == "x0":
                eps_hat = (x_t - torch.sqrt(acc_alpha_t) * model_out) / torch.sqrt(1 - acc_alpha_t)

            mean = (1 / torch.sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / torch.sqrt(1 - acc_alpha_t)) * eps_hat)             
            sigma = torch.sqrt(beta_t)
            x_t = mean + sigma * z

        return x_t 


    @torch.no_grad()
    def _sample_ddim(
        self,
        num_samples: int,
        sample_time_steps: int = 50,
        eta : float = 0,
        use_ema_model: bool = True
    ):
        """
            Generates samples using the DDIM (Denoising Diffusion Implicit Models) sampling method.

            This method performs deterministic or semi-deterministic sampling from the learned reverse diffusion process.
            It starts from pure Gaussian noise and iteratively denoises it using the model predictions, following the DDIM formulation.
            The parameter `eta` controls the stochasticity of the sampling: `eta=0` yields deterministic samples (pure DDIM),
            while `eta>0` introduces controlled noise similar to DDPM. The sample_time_steps controlls the computation time vs. quality 

            Parameters:
                num_samples (int):
                    Number of samples to generate.
                sample_time_steps (int):
                    Number of time steps used during sampling controlls the computation time vs. sampling quality. Defaults to 50.
                eta (float):
                    Noise scale parameter. If set to 0, the sampling is deterministic. Defaults to 0.

            Returns:
                torch.Tensor:
                    Generated samples of shape `(num_samples, *output_data_shape)`.

            Source:
                "Denoising Diffusion Implicit Models" by Jiaming Song, Chenlin Meng, Stefano Ermon. Link: https://arxiv.org/pdf/2010.02502
            """

        model, time_embed_mlp = self._get_active_model_eval(use_ema_model= use_ema_model)

        step_size = self.time_steps // sample_time_steps
        x_t = torch.randn((num_samples, *self.output_data_shape), device=self.device)

        for step in range(sample_time_steps):
            t = self.time_steps - step * step_size      # calculate our t from step size 
            t_next = max(t - step_size, 1)              # should not become 0 because we go from 1 to T time steps

            alpha_acc_t = self.acc_alphas[t - 1].reshape(-1, 1, 1, 1)                   # we need to reshape because we want dim [1, 1, 1, 1] and t - 1 because we index t from {1, ..., sample_time_steps} but list starts at 0
            alpha_acc_prev = self.acc_alphas[t_next - 1].reshape(-1, 1, 1, 1)    # same here 

            # if eta = 0 we have the denoising diffusion implicit model and sigma will be 0 => with this if we save computation 
            if eta != 0:
                sigma = eta * torch.sqrt((1 -  alpha_acc_prev) / (1 - alpha_acc_t)) * torch.sqrt(1 - alpha_acc_t / alpha_acc_prev)
                z = torch.randn_like(x_t) if step > 1 else torch.zeros_like(x_t)   # random part 
            else:
                sigma = torch.zeros_like(alpha_acc_t, device=self.device)
                z = torch.zeros_like(x_t)
            
            emb = self.raw_time_embeding(torch.full((num_samples,), t, device=self.device, dtype=torch.long))
            time_embedding = time_embed_mlp(emb) 

            model_out = model(x_t, time_embedding)
            if self.prediction_type == "eps":
                eps_hat = model_out
            elif self.prediction_type == "x0":
                eps_hat = (x_t - torch.sqrt(alpha_acc_t) * model_out) / torch.sqrt(1 - alpha_acc_t)

            x_t =   (
                        torch.sqrt(alpha_acc_prev) * 
                        ((x_t - torch.sqrt(1 - alpha_acc_t) * eps_hat) / torch.sqrt(alpha_acc_t)) +      # pred x0 part
                        torch.sqrt(1 - alpha_acc_prev - sigma**2) * eps_hat +                            # direction pointin to x0
                        sigma * z                                                                        # random part 
                    )

        return x_t 


    def _phi(self, x_t, eps_t, t, t_next):
        """

            Source:
                "PSEUDO NUMERICAL METHODS FOR DIFFUSION MODELS ON MANIFOLD" by Luping Liu, Yi Ren, Zhijie Lin & Zhou Zhao (EQ 11). Link: https://arxiv.org/pdf/2202.09778 
        """

        acc_alpha_t = self.acc_alphas[t - 1].reshape(-1, 1, 1, 1)                   # like this the eq will be cleaner 
        acc_alpha_t_next = self.acc_alphas[t_next - 1].reshape(-1, 1, 1, 1)       

        return (
                    (torch.sqrt(acc_alpha_t_next) / torch.sqrt(acc_alpha_t)) * x_t -
                    ((acc_alpha_t_next - acc_alpha_t) / (torch.sqrt(acc_alpha_t) * (torch.sqrt(1 - acc_alpha_t_next) * torch.sqrt(acc_alpha_t) + torch.sqrt((1 - acc_alpha_t) * acc_alpha_t_next)))) * eps_t
               )



    def _step_prk(self, x_t: torch.Tensor, t: int, t_next: int):
        """
        
        
        Source:
                "PSEUDO NUMERICAL METHODS FOR DIFFUSION MODELS ON MANIFOLD" by Luping Liu, Yi Ren, Zhijie Lin & Zhou Zhao (EQ 13). Link: https://arxiv.org/pdf/2202.09778 
        """
        model, time_embed_mlp = self._get_active_model_eval()
        
        t_half = (int(t - (t - t_next) / 2))

        emb = self.raw_time_embeding(torch.full((x_t.shape[0],), t, device=self.device, dtype=torch.long))
        t_embedding = time_embed_mlp(emb) 

        emb = self.raw_time_embeding(torch.full((x_t.shape[0],), t_half, device=self.device, dtype=torch.long))
        t_half_embeding = time_embed_mlp(emb) 
       
        emb = self.raw_time_embeding(torch.full((x_t.shape[0],), t_next, device=self.device, dtype=torch.long))
        t_next_embeding = time_embed_mlp(emb) 

        eps_1 = model(x_t, t_embedding)
        x_1 = self._phi(x_t, eps_1, t, t_half)

        eps_2 = model(x_1, t_half_embeding)
        x_2 = self._phi(x_t, eps_2, t, t_half)

        eps_3 = model(x_2, t_half_embeding)
        x_3 = self._phi(x_t, eps_3, t, t_next)

        eps_4 = model(x_3, t_next_embeding)
        eps_next = (eps_1 + 2 * eps_2 + 2 * eps_3 + eps_4) / 6

        x_next = self._phi(x_t, eps_next, t, t_next)

        return x_next, eps_1


    def _step_plms(self, x_t: torch.Tensor, t: int, t_next: int, eps_buffer: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            One PLMS update 

            Source:
                "PSEUDO NUMERICAL METHODS FOR DIFFUSION MODELS ON MANIFOLD" by Luping Liu, Yi Ren, Zhijie Lin & Zhou Zhao (EQ 12). Link: https://arxiv.org/pdf/2202.09778 
        """
        model, time_embed_mlp = self._get_active_model_eval()
    
        emb = self.raw_time_embeding(torch.full((x_t.shape[0],), t, device=self.device, dtype=torch.long))
        t_embedding = time_embed_mlp(emb) 

        eps_t = model(x_t, t_embedding)
        eps_next = (55 * eps_t - 59 * eps_buffer[-1] + 37 * eps_buffer[-2] - 9 * eps_buffer[-3]) / 24.0
        x_next = self._phi(x_t, eps_next, t, t_next)

        return x_next, eps_t


    @torch.no_grad()
    def _sample_pndm(
        self,
        num_samples: int,
        sample_time_steps: int = 50
    ) -> torch.Tensor:
        """
            Generates samples using the pndm algorithm (PSEUDO NUMERICAL METHODS FOR DIFFUSION MODELS) sampling method.

            Parameters:
                num_samples (int):
                    Number of samples to generate.
                sample_time_steps (int):
                    Number of time steps used during sampling controlls the computation time vs. sampling quality. Defaults to 50.

            Returns:
                torch.Tensor:
                    Generated samples of shape `(num_samples, *output_data_shape)`.

            Source:
                "PSEUDO NUMERICAL METHODS FOR DIFFUSION MODELS ON MANIFOLD" by Luping Liu, Yi Ren, Zhijie Lin & Zhou Zhao. Link: https://arxiv.org/pdf/2202.09778
        """
        step_size = max(1, self.time_steps // sample_time_steps)
        
        x_t = torch.randn((num_samples, *self.output_data_shape), device=self.device)
        eps_buffer = ["", "", ""]   # init this with 3 elements becasue then we can append an pop in every step and always have 3 elements 

     
        ts = list(range(self.time_steps, 0, -step_size))
        if ts[-1] != 1:
            ts.append(1)
        
        counter = 0

        for t, t_next in zip(ts[:-1], ts[1:]):
            if counter < 3:
                counter += 1
                x_t, e_t = self._step_prk(x_t=x_t, t=t, t_next=t_next)
            else:
                x_t, e_t = self._step_plms(x_t=x_t, t=t, t_next=t_next, eps_buffer=eps_buffer)

            eps_buffer.append(e_t)
            eps_buffer.pop(0)

        return x_t


    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        batch_size: int = 32,
        sampler: str = "ddim", # "ddim" or "ddpm" or "pndm"
        sample_time_steps: int = 50,
        use_ema_model: bool = False,
        eta : float = 0
    ) -> torch.Tensor:
        """
                Wrapper for the sampling function, supporting different samplers (DDIM, DDPM, PNDM).
                DDIM is typically faster than DDPM.

                The method generates `num_samples` outputs in a **batch-wise manner** to avoid memory issues
                when sampling large numbers of images. All batches are combined into a single tensor.

                Parameters
                    num_samples (int):
                        Total number of samples to generate.
                    batch_size (int):
                        Number of samples per batch. Sampling is performed in chunks of this size.
                    sampler (str):
                        Sampling algorithm to use. Options:
                            - "ddim" : Deterministic Diffusion Implicit Model (fast)
                            - "ddpm" : Denoising Diffusion Probabilistic Model
                            - "pndm" : Pseudo Numerical Methods for Diffusion Models
                    sample_time_steps (int):
                        Number of timesteps for DDIM sampling (only relevant if sampler="ddim").
                    eta (float):
                        DDIM noise parameter (controls stochasticity, only relevant for DDIM).

                Returns
                    torch.Tensor:
                        A tensor containing all generated samples with shape:
                        (num_samples, *output_data_shape).
                        The samples are generated in batches and concatenated into one tensor.
        """

        self.eval()

        if sampler == "ddpm":
            sample_batch_fn = lambda bs: self._sample_ddpm(num_samples=bs, use_ema_model= use_ema_model)
        elif sampler == "ddim":
            sample_batch_fn = lambda bs: self._sample_ddim(num_samples=bs, sample_time_steps=sample_time_steps, eta=eta, use_ema_model= use_ema_model)
        elif sampler == "pndm":
            raise ValueError("pndm dosent work right know please use 'ddim' or 'ddpm'")
            sample_batch_fn = lambda bs: self._sample_pndm(num_samples=bs, sample_time_steps=sample_time_steps, use_ema_model= use_ema_model)
        else:
            raise ValueError(f"Unknown sampler: {sampler}. Use 'ddim', 'ddpm', or 'pndm'.")

        num_batches = math.ceil(num_samples / batch_size)        
        all_samples = torch.empty((num_samples, *self.output_data_shape), device=self.device)

        start_idx = 0
        for _ in range(num_batches):
            current_batch_size = min(batch_size, num_samples - start_idx)
            batch = sample_batch_fn(current_batch_size)
            all_samples[start_idx : start_idx + current_batch_size] = batch
            start_idx += current_batch_size

        return all_samples


    def _get_active_model_train(self):
        use_comp = (
            self.comp_model is not None
            and self.comp_time is not None
            and self.training
            and torch.is_grad_enabled()
        )
        return (self.comp_model, self.comp_time) if use_comp else (self.model, self.time_embed_mlp)


    def _get_active_model_eval(self, use_ema_model: bool = True):
        if use_ema_model and self.use_ema and self.ema_model is not None and self.ema_embed is not None:
            if self.comp_ema_model is not None and self.comp_ema_time is not None:
                return self.comp_ema_model, self.comp_ema_time
            return self.ema_model, self.ema_embed

        if self.comp_model is not None and self.comp_time is not None:
            return self.comp_model, self.comp_time
        
        return self.model, self.time_embed_mlp



    def train_step(
        self, 
        x0: torch.Tensor, 
        batch_size: int = 64
    ):
        """
            Performs a single training step using the selected diffusion objective.

            Depending on `self.mode`, this method will either:
            - Predict complete noise ε (DDPM objective), or
            - Predict the original sample x₀ (alternative parameterization).

            Parameters
                x0 (torch.Tensor):
                    Batch of original data samples.
                batch_size : int
                    Number of samples in the batch.

            Returns
                torch.Tensor
                    Scalar loss value for the current training step.
                dict
                    Dictionary of logging metrics (e.g., BCE loss, MSE loss).
        """
        if self.prediction_type == "eps":
            return self._train_step_eps(x0, batch_size)
        elif self.prediction_type == "x0":
            return self._train_step_x0(x0, batch_size)
        else:
            raise ValueError(f"Unsupported prediction_type suports 'eps' and 'x0' but is: {self.prediction_type}")


    def _train_step_eps(self, x0: torch.Tensor, batch_size: int):
        """
            Performs a training step using the DDPM objective (predicting complete noise ε).
            Source: Algorithm 1 from https://arxiv.org/abs/2006.11239

            Parameters:
          
                x0 (torch.Tensor):
                    Batch of original data samples.
                batch_size (int):
                    Number of samples in the batch.

            Returns
    
                torch.Tensor
                    Scalar loss value for the current training step.
                dict
                    Dictionary of logging metrics.
        """
        model, time_embed_mlp = self._get_active_model_train()

        t = torch.randint(1, self.time_steps + 1, (batch_size,), device=self.device)
        
        time_embed = self.raw_time_embeding(t)
        time_embed = time_embed_mlp(time_embed)

        acc_alpha_t = self.acc_alphas[t - 1].reshape(-1, 1, 1, 1)

        epsilon = torch.randn_like(x0)

        x_t = torch.sqrt(acc_alpha_t) * x0 + torch.sqrt(1 - acc_alpha_t) * epsilon      # Compute noisy input x_t

        epsilon_pred = model(x_t, time_embed)

        loss, metrics = self.loss_func(epsilon, epsilon_pred)
        return loss, metrics


    def _train_step_x0(self, x0: torch.Tensor, batch_size: int):
        """
            Performs a training step using x₀ prediction (alternative DDPM parameterization).

            This method predicts the original sample x₀ instead of noise ε.
            The loss is weighted by the Signal-to-Noise Ratio (SNR) for stability.

            Parameter:
                x0 (torch.Tensor):
                    Batch of original data samples.
                batch_size (int):
                    Number of samples in the batch.
            Returns
                torch.Tensor
                    Scalar loss value for the current training step.
                dict
                    Dictionary of logging metrics.
        """
        t = torch.randint(1, self.time_steps + 1, (batch_size,), device=self.device)

        model, time_embed_mlp = self._get_active_model_train()

        time_embed = self.raw_time_embeding(t)
        time_embed = time_embed_mlp(time_embed)

        acc_alpha_t = self.acc_alphas[t - 1].reshape(-1, 1, 1, 1)

        epsilon = torch.randn_like(x0)

        x_t = torch.sqrt(acc_alpha_t) * x0 + torch.sqrt(1 - acc_alpha_t) * epsilon
        x0_pred = model(x_t, time_embed)

        snr_weight = (acc_alpha_t / (1 - acc_alpha_t))

        loss, metrics = self.loss_func(x0, x0_pred, batch_weights = snr_weight)
        return loss, metrics


    def train_model(
        self,
        sample_batch: Callable[[int], torch.Tensor],
        num_train_batches: int = 100_000,
        batch_size: int = 64,
        log_train_loss_per_batch: int = 100,
        log_test_loss_per_batch: int = 1000,
        num_test_batches_log: int = 10,
        sample_test_batch: Callable[[int], torch.Tensor] = None,
        checkpoint_per_batch: int = 25_000,
        checkpoint_path: str = None,
        resume_from_checkpoint: str = None,
        max_grad_norm: float = 1.0,
        lr: float = 2e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.01,
        use_scheduler: bool = False,
        tensorboard_writer: SummaryWriter = None,
        loggin_start_batch: int = 0
    ) -> pd.DataFrame:

        # Calculate spacing for console output
        max_batch_width = len(str(num_train_batches))
        max_example_width = len(str(num_train_batches * batch_size))

        # DataFrame to store training progress
        train_info_df = pd.DataFrame(columns=["Batch", "Batch_Size", "Train_Loss", "Test_Loss", "lr", "grad_norm"]).astype({"Batch": "int64", "Batch_Size": "int64", "Train_Loss": "float64", "Test_Loss": "float64", "lr": "float32", "grad_norm": "float32"})

        # Optimizer
        if not hasattr(self, "optimizer") or self.optimizer is None:
            self.optimizer = torch.optim.AdamW(self.trainable_params, lr=lr, betas=betas, weight_decay=weight_decay)

        # Scheduler
        if use_scheduler:
            if not hasattr(self, "scheduler") or self.scheduler is None:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,T_max=num_train_batches,eta_min=1e-5)
        else:
            if not hasattr(self, "scheduler") or self.scheduler is None:
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,lr_lambda=lambda step: 1.0)

        # Resume training
        start_batch = 1
        if resume_from_checkpoint:
            checkpoint = torch.load(resume_from_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_batch = checkpoint.get("step", 1) + 1
            print(f"Resuming training from batch {start_batch}")

        # Progress bar
        with tqdm(total=num_train_batches, desc="Training", leave=True) as pbar:
            pbar.update(start_batch - 1)

            for num_batch in range(start_batch, num_train_batches + 1):

                # ---------------------- TRAIN STEP ----------------------
                self.train()
                batch = sample_batch(batch_size).to(self.device, non_blocking=True)

                loss, logging_metrics = self.train_step(batch, batch_size=batch_size)

                self.optimizer.zero_grad()
                loss.backward()

                # grad norm calc and clip
                effective_max_norm = max_grad_norm if (max_grad_norm is not None and max_grad_norm > 0) else float("inf")
                grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable_params, effective_max_norm).item()

                self.optimizer.step()
                self.scheduler.step()
                self.ema_update()
                pbar.update(1)

                # ---------------------- TEST STEP -----------------------
                test_loss = None
                test_metrics_accum = {}

                if sample_test_batch and num_batch % log_test_loss_per_batch == 0:
                    self.eval()
                    total_loss = 0.0

                    with torch.no_grad():
                        for _ in range(num_test_batches_log):
                            test_batch = sample_test_batch(batch_size).to(self.device)
                            t_loss, t_metrics = self.train_step(test_batch, batch_size=batch_size)
                            total_loss += t_loss.item()

                            if t_metrics:
                                for k, v in t_metrics.items():
                                    test_metrics_accum[k] = test_metrics_accum.get(k, 0.0) + float(v)

                    test_loss = total_loss / num_test_batches_log

                    if test_metrics_accum:
                        for k in test_metrics_accum:
                            test_metrics_accum[k] /= num_test_batches_log

                # ---------------------- LOGGING ------------------
                train_info_df, _ = self._log_step(
                    tensorboard_writer=tensorboard_writer,
                    train_info_df=train_info_df,
                    num_batch=num_batch,
                    loggin_start_batch=loggin_start_batch,
                    batch_size=batch_size,
                    loss_current=float(loss.item()),
                    logging_metrics=logging_metrics,
                    test_loss=test_loss,
                    test_metrics_accum=test_metrics_accum if test_metrics_accum else None,
                    log_train_loss_per_batch=log_train_loss_per_batch,
                    log_test_loss_per_batch=log_test_loss_per_batch,
                    max_batch_width=max_batch_width,
                    max_example_width=max_example_width,
                    grad_norm = grad_norm
                )

                # --------------------- CHECKPOINTING ---------------------
                if checkpoint_path and num_batch % checkpoint_per_batch == 0:
                    os.makedirs(checkpoint_path, exist_ok=True)
                    ckpt_file = os.path.join(checkpoint_path, f"model_step_{num_batch}.pt")
                    self.save_checkpoint(num_batch, ckpt_file)

        return train_info_df


    @torch._dynamo.disable
    @torch.no_grad()
    def _log_step(
        self,
        *,
        tensorboard_writer: SummaryWriter,
        train_info_df: pd.DataFrame,
        num_batch: int,
        loggin_start_batch: int,
        batch_size: int,
        loss_current: float,
        logging_metrics: dict,
        test_loss: float,
        test_metrics_accum: dict,
        log_train_loss_per_batch: int,
        log_test_loss_per_batch: int,
        max_batch_width: int,
        max_example_width: int,
        flush_every: int = 1000,
        grad_norm: float = 0,
    ) -> tuple[pd.DataFrame, str]:
        """
            Handles all logging responsibilities for a single training step:
            - TensorBoard scalar logging (Train/Test metrics, LR, grad norm)
            - Optional TensorBoard text summary
            - Console output via tqdm.write
            - Adding rows to the training info DataFrame

            Returns:
                (updated DataFrame, log string)
        """

        # Compute global step for TensorBoard
        global_step = num_batch + loggin_start_batch

        # ------------------------------------------------------------------
        #                        TENSORBOARD LOGGING
        # ------------------------------------------------------------------

        if tensorboard_writer:
            tensorboard_writer.add_scalar("Loss/train", float(loss_current), global_step)   # Log training loss every step

            # Log additional training metrics
            if logging_metrics:
                for k, v in logging_metrics.items():
                    tensorboard_writer.add_scalar(f"Train/{k}", float(v), global_step)

            # Log optimizer information
            tensorboard_writer.add_scalar("Opt/lr", float(self.scheduler.get_last_lr()[0]), global_step)
            tensorboard_writer.add_scalar("Opt/grad_norm", float(grad_norm), global_step)

        # Log test metrics if they exist
        if tensorboard_writer and test_loss is not None:
            tensorboard_writer.add_scalar("Loss/test", float(test_loss), global_step)

            if test_metrics_accum:
                for k, v in test_metrics_accum.items():
                    tensorboard_writer.add_scalar(f"Test/{k}", float(v), global_step)

        # ------------------------------------------------------------------
        #                        CONSOLE LOG STRING
        # ------------------------------------------------------------------
       
        show_train_log = (log_train_loss_per_batch and num_batch % log_train_loss_per_batch == 0)
        show_test_log = (log_test_loss_per_batch and num_batch % log_test_loss_per_batch == 0 and test_loss is not None)

        log_str = (
            f"Batch {num_batch + loggin_start_batch:>{max_batch_width}} || "
            f"Example {(num_batch + loggin_start_batch) * batch_size:>{max_example_width}} || "
        )

        log_str += f"Train Loss = {loss_current:.6f} || " if show_train_log else "Train Loss = N/A      || "
        log_str += f"Test Loss = {test_loss:.6f} || " if show_test_log else "Test Loss = N/A      || "

        if logging_metrics and show_train_log:
            log_str += " || " + " || ".join([f"Train {k}={float(v):.4f}" for k, v in logging_metrics.items()])

        if test_metrics_accum and show_test_log:
            log_str += " || " + " || ".join([f"Test {k}={float(v):.4f}" for k, v in test_metrics_accum.items()])

        if show_train_log or show_test_log:
            tqdm.write(log_str)

        # ------------------------------------------------------------------
        #                      DATAFRAME ROW APPEND
        # ------------------------------------------------------------------
        if show_train_log or show_test_log:
            row = {
                "Batch": num_batch + loggin_start_batch,
                "Train_Loss": loss_current if show_train_log else np.nan,
                "Test_Loss": test_loss if show_test_log else np.nan,
                "Batch_Size": batch_size,
                "lr": self.scheduler.get_last_lr()[0],
                "grad_norm": grad_norm
            }

            # Add Train/… metrics
            if logging_metrics:
                for k, v in logging_metrics.items():
                    row[f"Train_{k}"] = float(v)

            # Add Test/… metrics
            if test_metrics_accum:
                for k, v in test_metrics_accum.items():
                    row[f"Test_{k}"] = float(v)

            train_info_df = pd.concat([train_info_df, pd.DataFrame([row])], ignore_index=True)

        # ------------------------------------------------------------------
        #             PERIODIC TENSORBOARD FLUSH (avoids data loss)
        # ------------------------------------------------------------------
        if tensorboard_writer and (num_batch % flush_every == 0):
            tensorboard_writer.flush()

        return train_info_df, log_str


    @torch.no_grad()
    def ema_update(self):
        if not self.use_ema or self.ema_model == None:
            return
        
        model_state = self.model.state_dict()
        ema_model_state = self.ema_model.state_dict()

        for k in model_state.keys():
            ema_model_state[k].mul_(self.ema_decay).add_(model_state[k], alpha= 1.0- self.ema_decay)

        time_state = self.time_embed_mlp.state_dict()
        ema_time_state = self.ema_embed.state_dict()

        for k in time_state.keys():
            ema_time_state[k].mul_(self.ema_decay).add_(time_state[k], alpha= 1.0- self.ema_decay)


    def summary(self, print_model_structure: bool = False) -> str:
        """
            Returns a detailed summary of the diffusion model, including:
            - Function approximator architecture and parameter count
            - Time embedding MLP architecture and parameter count
            - Total number of trainable parameters
            - Key configuration details (device, prediction type, schedule type, etc.)
        """

        def count_parameters(module: torch.nn.Module) -> int:
            """Counts the number of trainable parameters in a given module."""
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        # Count parameters for each component
        model_params = count_parameters(self.model)
        time_embed_params = count_parameters(self.time_embed_mlp)
        total_params = model_params + time_embed_params

        # Build the summary string
        summary_str = "\n" + "=" * 60 + "\n"
        summary_str += "DIFFUSION MODEL SUMMARY\n"
        summary_str += "=" * 60 + "\n\n"

        # General configuration
        summary_str += f"Device: {self.device}\n"
        summary_str += f"Prediction Type: {self.prediction_type}\n"
        summary_str += f"Schedule Type: {self.scedule_type}\n"
        summary_str += f"Time Steps: {self.time_steps}\n"
        summary_str += f"Sinusoidal Time Embedding Dim: {self.sinus_time_embeding_dim}\n"
        summary_str += f"MLP Time Embedding Dim: {self.time_embeding_dim}\n"
        summary_str += f"EMA Enabled: {self.use_ema}\n\n"
        summary_str += "-" * 50 + "\n"
        summary_str += f"Total Trainable Parameters: {total_params:,}\n"
        summary_str += "-" * 50 + "\n\n"

        summary_str += "-" * 50 + "\n"
        summary_str += f"Total Forward Flops: {self.compute_forward_flops():,}\n"
        summary_str += "-" * 50 + "\n\n"

        # Function Approximator
        summary_str += "=" * 60 + "\n"
        summary_str += "Function Approximator\n"
        summary_str += "=" * 60 + "\n"
        if print_model_structure: summary_str += f"{str(self.model)}\n"
        summary_str += "-" * 50 + "\n"
        summary_str += f"Number of Parameters: {model_params:,}\n"
        summary_str += "-" * 50 + "\n\n"

        # Time Embedding MLP
        summary_str += "=" * 60 + "\n"
        summary_str += "Time Embedding MLP\n"
        summary_str += "=" * 60 + "\n"
        if print_model_structure: summary_str += f"{str(self.time_embed_mlp)}\n"
        summary_str += "-" * 50 + "\n"
        summary_str += f"Number of Parameters: {time_embed_params:,}\n"
        summary_str += "-" * 50 + "\n\n"

        return summary_str


    @torch.no_grad()
    def compute_forward_flops(self):
        """
            Computes the FLOPs of one forward pass of the diffusion model:
            (x_t, time_emb) -> model output.

            Parameters
                input_resolution : (H, W)
                    Optional. If None, uses the model's output_data_shape.

            Returns
                total_flops : int
                    Total number of FLOPs for one forward pass.
        """
        torch._dynamo.reset()
        torch._dynamo.disable()

        # Prepare dummy inputs
        x_dummy = torch.randn([1] + self.output_data_shape, device=self.device)
        t_dummy = torch.randint(1, self.time_steps, (1,), device=self.device)

        t_emb = t_dummy.type(torch.float32)[:, None] * self.freq[None, :]
        t_emb = torch.concat([torch.sin(t_emb), torch.cos(t_emb)], dim = 1)

        class Wrapper(torch.nn.Module):
            def __init__(self, function_approximator, time_embed):
                super().__init__()
                self.function_approximator = function_approximator
                self.time_embed = time_embed

            def forward(self, x, t):
                return self.function_approximator(x, self.time_embed(t))

        wraper = Wrapper(self.model, self.time_embed_mlp)
        flops = FlopCountAnalysis(wraper.to(self.device), (x_dummy, t_emb)) 
    
        return flops.total()


    def save_checkpoint(self, num_batch, path="checkpoints/diffusion_model.pt") -> None:
        """
            Save the model weights + optimizer state + num_batch in a file.

            Parameters
                path (str):
                    File path for saving (e.g., 'unet.pt').
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            "model_state": self.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "trained_batches": num_batch,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state"] = self.scheduler.state_dict()

        checkpoint["rng_state"] = torch.get_rng_state()
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()

        torch.save(checkpoint, path)


    def load_checkpoint(self, path: str, map_location=None) -> int:
        """
            Load model weights + optimizer + scheduler + RNG state.

            Returns
            -------
            trained_batches : int
                Number of batches the model was already trained on.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(
            path,
            map_location=map_location or self.device
        )

        self.load_state_dict(checkpoint["model_state"])

        if "optimizer_state" in checkpoint and hasattr(self, "optimizer"):
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            except ValueError as e:
                print("[WARN] Could not load optimizer state:", e)

        if "scheduler_state" in checkpoint and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            except ValueError as e:
                print("[WARN] Could not load scheduler state:", e)

        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"])

        if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
            except RuntimeError as e:
                print("[WARN] Could not restore CUDA RNG state:", e)

        trained_batches = checkpoint.get("trained_batches", 0)

        return trained_batches


    def save(self, path: str) -> None:
        """
            Save the model weights to a file.

            Parameters
                path (str):
                    File path for saving (e.g., 'unet.pt').
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)


    def load(self, path: str, map_location: Optional[str] = None) -> None:
        """
            Load the model weights from a file.

            Parameters
                path (str):
                    File path to load from.
                map_location (Optional[str]):
                    device mapping
        """
        state = torch.load(path, map_location=map_location or "cpu")
        self.load_state_dict(state)