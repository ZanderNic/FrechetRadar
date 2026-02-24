# std lib imports
import math

# 3 party import
import torch

# projekt imports


class NoiseSchedule:
    def __init__(self, time_steps: int, schedule_type: str = "sigmoid"):
        self.time_steps = time_steps
        self.schedule_type = schedule_type
        self.betas = self._create_schedule()
        self.alphas = 1 - self.betas
        self.acc_alphas = torch.cumprod(self.alphas, dim=0)

        self.check_schedule_parameters()    # check all parms 

    
    def _create_schedule(self):
        if self.schedule_type == "linear":
            return torch.linspace(1e-4, 0.02, self.time_steps)  # from paper https://arxiv.org/pdf/2006.11239
        elif self.schedule_type == "cosine":
            return self._cosine_schedule()                      # from paper from https://arxiv.org/pdf/2102.09672
        elif self.schedule_type == "sigmoid":
            return torch.sigmoid(torch.linspace(-18, 10, self.time_steps)) * (3e-1 - 1e-5) + 1e-5
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")


    def _cosine_schedule(self):
        """
            cosine Schedule for Denoising Diffusion Provalistic Models from https://arxiv.org/pdf/2102.09672
        """
        s = 0.008
        steps = torch.arange(self.time_steps + 1, dtype=torch.float32)
        f = torch.cos(((steps / self.time_steps + s) / (1 + s)) * math.pi * 0.5) ** 2
        alphas = f / f[0]
        betas = 1 - (alphas[1:] / alphas[:-1])
        return torch.clamp(betas, 1e-5, 0.99999)


    def check_schedule_parameters(self):
        """
        Validates betas, alphas, and acc_alphas to ensure the diffusion schedule is consistent.
        """
        if self.betas.ndim != 1:
            raise ValueError(f"betas must be 1D, got shape {self.betas.shape}")

        if self.alphas.ndim != 1:
            raise ValueError(f"alphas must be 1D, got shape {self.alphas.shape}")

        if self.acc_alphas.ndim != 1:
            raise ValueError(f"acc_alphas must be 1D, got shape {self.acc_alphas.shape}")

        if not torch.isfinite(self.betas).all():
            raise ValueError("betas contain NaN or Inf")

        if not torch.isfinite(self.alphas).all():
            raise ValueError("alphas contain NaN or Inf")

        if not torch.isfinite(self.acc_alphas).all():
            raise ValueError("acc_alphas contain NaN or Inf")

        bet_min, bet_max = self.betas.min().item(), self.betas.max().item()
        if not ((self.betas > 0).all() and (self.betas < 1).all()):
            raise ValueError(f"betas must be in (0,1). Range: [{bet_min}, {bet_max}]")

        alp_min, alp_max = self.alphas.min().item(), self.alphas.max().item()
        if not ((self.alphas > 0).all() and (self.alphas < 1).all()):
            raise ValueError(f"alphas must be in (0,1). Range: [{alp_min}, {alp_max}]")

        acc_min, acc_max = self.acc_alphas.min().item(), self.acc_alphas.max().item()
        if not ((self.acc_alphas > 0).all() and (self.acc_alphas <= 1).all()):
            raise ValueError(f"acc_alphas must be in (0,1]. Range: [{acc_min}, {acc_max}]")
        
        if not torch.all(self.acc_alphas[:-1] >= self.acc_alphas[1:]):
            raise ValueError("acc_alphas must be monotonically decreasing")

        reconstructed = torch.cumprod(self.alphas, dim=0)
        diff = torch.max(torch.abs(reconstructed - self.acc_alphas)).item()

        if diff > 1e-6:
            raise ValueError(
                f"Inconsistent schedule: acc_alphas != cumprod(alphas). "
                f"Max difference: {diff:.6e}. "
                f"This usually means you clipped acc_alphas incorrectly or modified only one part of the schedule."
            )