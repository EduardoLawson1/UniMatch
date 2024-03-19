import os
import subprocess

import torch
import torch.distributed as dist


print(torch.cuda.is_available())
#teste
cuda2 = torch.device('cuda:0')



def setup_distributed(backend="nccl", port=None):
    """AdaHessian Optimizer
    Lifted from https://github.com/BIGBALLON/distribuuuu/blob/master/distribuuuu/utils.py
    Originally licensed MIT, Copyright (c) 2020 Wei Li
    """
    num_gpus = torch.cuda.device_count()
    print(f"\n\n\n\nNumeros de gpu: {num_gpus}\n\n\n\n")
    
    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        print(f'rank = {rank} \n gpu ={num_gpus} \n\n\n\n')
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "10685"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f'rank = {rank} \n gpu ={num_gpus} \n\n\n\n')
    #teste
    #torch.cuda.set_device(rank % num_gpus)
    #torch.cuda.set_device(num_gpus)
    torch.cuda.set_device(cuda2)
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size
