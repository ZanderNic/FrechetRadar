# std lib imports


# 3 party import
import torch

# projekt imports
from RadarDataGen.Metrics.frechet_distance import frechet_distance_generator

# Test Params ########################################

batch_size = 64
num_batches = 10
feature_dim = 5
device = "cpu"

######################################################

# dummy random generator 
def constant_generator(value: torch.Tensor, num_batches: int):
    for _ in range(num_batches):
        yield value.clone()

def random_generator(batch_size: int, feature_dim: int, num_batches: int, seed: int = 42):
    torch.manual_seed(seed)
    for _ in range(num_batches):
        yield torch.randn(batch_size, feature_dim)


# As feature extractor we take a dummy extractor that dose nothing 
def identity_extractor(x: torch.Tensor) -> torch.Tensor:
    return x


if __name__ == "__main__":
    
    # we need to get fd == 0  
    data = torch.randn(batch_size, feature_dim)
    gen1 = constant_generator(data, num_batches)
    gen2 = constant_generator(data, num_batches)

    fid_0 = frechet_distance_generator(gen1, gen2, identity_extractor, feature_dim, device=device)  
    print(f"Fid should be 0 and is: {fid_0}")


    # we neet to get a fd >= 0
    gen1 = constant_generator(torch.ones(batch_size, feature_dim), num_batches)
    gen2 = constant_generator(torch.zeros(batch_size, feature_dim), num_batches)

    fid_1 = frechet_distance_generator(gen1, gen2, identity_extractor, feature_dim, device=device)
    print(f"Fid should be > 0 and is: {fid_1}")
