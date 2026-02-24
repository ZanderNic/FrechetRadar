# std lib imports

# 3 party import
import numpy as np
import torch

# projekt imports
from RadarDataGen.Models.DiffusionModell.diff_model import DiffusionModel
from RadarDataGen.Models.UNet.u_net import U_Net


def sample_constant_batch(batch_size, shape, device):
    return torch.ones((batch_size, *shape), device=device) * 0.5

   
if __name__ == "__main__":

    ###  Setting  #####################
    output_data_shape=(1, 16, 16)
    time_embedding_dim = 256
    device= "cuda" if torch.cuda.is_available() else "cpu"
    ###################################

    u_net = U_Net(
        input_chanels = output_data_shape[0],
        time_embedding_dim= time_embedding_dim,
        channels_per_level= [64, 128, 256, 512],
        attention_levels= [0, 0, 1, 1],
        resnet_blocks_per_depth= 2,
        device=device
    )

    model = DiffusionModel(
        function_approximator=u_net,
        output_data_shape=output_data_shape,
        time_steps=1_000,
        mlp_time_embeding_dim=time_embedding_dim,
        sinus_time_embeding_dim=128,
        schedule_type="cosine",
        device=device
    )

    histroy = model.train_model(
        sample_batch=lambda bs: sample_constant_batch(bs, model.output_data_shape, device=device),
        num_train_batches= 10, #100_000,
        batch_size=256,
        log_train_loss_per_batch=10,
        log_test_loss_per_batch=20,
        sample_test_batch=lambda bs: sample_constant_batch(bs, model.output_data_shape, device=device),
        # checkpoint_path="./tests/checkpoints/",
        # checkpoint_per_batch=250
    )

    # model.load("./tests/model_test.pt" , map_location=device)
    # #model.save("./tests/model_test.pt")

    samples_ddpm = model.sample(1, sampler="ddpm")
    samples_ddim = model.sample(1, sampler="ddim", sample_time_steps=250, eta=0)    # if we set eta to 1 we have ddpm here to 
    #samples_pndm = model.sample(1, sampler="pndm", sample_time_steps=250)

    print("Should should look like torch.ones((shape)) * 0.5: \n", torch.ones((output_data_shape)) * 0.5)
    print()
    print("Sample from DDPM looks like: \n", samples_ddpm)
    print()
    print("Sample from DDIM  looks like: \n", samples_ddim)
    print()
    #print("Sample from PNDM looks like: \n", samples_pndm)
