# std lib imports

# 3 party import
import torch

# projekt imports


class OnlineStats():
    """
        Updates the running mean and covariance using the Chan-Golub-LeVeque algorithm 
        (a batched extension of Welford's method).

        This method processes a batch of observations and incrementally updates:
            - The global mean vector
            - The accumulated sum of squares (M2), which is later used to compute the covariance matrix

        Args:
            batch (torch.Tensor): A 2D tensor of shape (batch_size, feature_dim) containing the new observations.

        Notes:
            - The algorithm is numerically stable for streaming data.
            - The covariance matrix is not computed here; it is derived later from M2 and the sample count.
            - The batch is automatically moved to the configured device and dtype.
    """

    
    def __init__(
        self, 
        feature_dim: int,
        dtype: torch.dtype = torch.float64,
        device : str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.feature_dim = feature_dim
        self.mean, self.m2, self.num_samples = torch.zeros((1, feature_dim), device=device, dtype=dtype), torch.zeros((feature_dim, feature_dim), device=device, dtype=dtype), 0
        self.device = device

    def update(
        self,
        batch : torch.Tensor,  
    ):
        """
            Updates the mean and the cov by using Chan-Golub-LeVeque algorithm (Batch Welford algorithm)
        """
        if batch.numel() == 0:      
            return  
        
        batch = batch.to(self.device)
        B = batch.shape[0]
       
        batch_mean = batch.mean(dim=0, keepdim=True)
        batch_centered = batch - batch_mean 
        batch_cov = batch_centered.T @ batch_centered
       
        if self.num_samples == 0:
            self.mean = batch_mean
            self.m2 = batch_cov
            self.num_samples += B
            return
        
        num_prev  = self.num_samples
        self.num_samples += batch.shape[0]
        delta = batch_mean - self.mean

        self.mean = self.mean + delta * (B/self.num_samples)

        self.m2 = (
            self.m2 +
            batch_cov +
            delta.T @ delta * (num_prev * B / self.num_samples) 
        )


    def get_mean_cvar(
        self,
        unbiased: bool = True
    ):    
        """
            Returns the current mean vector and sample covariance matrix.

            Uses the accumulated statistics to compute the unbiased sample covariance matrix
            (divided by `num_samples - 1`). Raises an error if the number of samples is too low
            to compute a valid covariance matrix.

            Returns:
                mean (torch.Tensor): The current mean vector of shape (1, feature_dim).
                cov (torch.Tensor): The sample covariance matrix of shape (feature_dim, feature_dim).

            Raises:
                ValueError: If the number of samples is less than `max(feature_dim, 2)`.
        """
        if (self.num_samples <= self.feature_dim):
            raise ValueError(f"Have seen {self.num_samples} samples but to calculate covarianz need at least {self.feature_dim} (if cov should be inv else set this to self.num_samples < 2)")

        return self.mean, self.m2 / ((self.num_samples - 1) if unbiased else self.num_samples)
    
    
    def deepcopy(self):
        """
            Returns a copy of the Online Stats Instance  
        """
        copy = OnlineStats(self.feature_dim, device="cpu")
        copy.mean = self.mean.cpu().clone()
        copy.m2 = self.m2.cpu().clone()
        copy.num_samples = self.num_samples
        copy.to(self.device)
        return copy


    def to(self, device):
        """
            Moves OnlineStats class to device ("cpu", "cuda", ...)
        """
        self.mean = self.mean.to(device)
        self.m2 = self.m2.to(device)
        self.device = device
