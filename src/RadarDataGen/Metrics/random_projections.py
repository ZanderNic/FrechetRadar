# std lib imports

# 3 party import
import torch

# projekt imports

class RandomProjektions(torch.nn.Module):
    """
        A non trainable Module that is based on the paper "Near-Optimal Signal Recovery From Random Projections: Universal Encoding Strategies?" from Emmanuel J. Candes
        that shows that it is possible to recover a d dimensional signal x that is k sparse meeaning that out of the d dimensions there are only k with k << d non zero elements
        with high precission from only k random linear measurements. Therefore we can use a Matrix  M that has the dim d x k with random elements and y = M * x  
        with high precission with y having a dim of k so is mutch smaller than x.
    """
    
    def __init__(
        self,
        data_dim: int, 
        feature_dim: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
            Here we init our sensing matrix M with gaussian random variables and the dim of  data_dim x feature_dim so we can multiply our data vector to it and get the 
            not exact but goof enoth feature dim representation of our input dim. 
        """
        super().__init__()
        self.dim_ = feature_dim
        self.sensing_matrix = torch.randn(
            (data_dim, feature_dim),
            device=device, dtype=torch.float32
        )

    def forward(
        self,
        x: torch.Tensor
    ):
        """
            Applies the random projection to a batch of input vectors.
            Input: x of shape (batch_size, data_dim)
            Output: y of shape (batch_size, feature_dim)
        """

        x = x.reshape(x.size(0), -1)  # ensure (batch, data_dim)

        with torch.no_grad():
            y = x @ self.sensing_matrix

        return y

