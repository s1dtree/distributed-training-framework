import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run_dist_test(rank, world_size):
    """
    Function to be run by each process, demonstrating initialization
    & basic all-reduce collective operation.
    """

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    tensor = torch.tensor([float(rank + 1)])
    print(f"Process {rank}: Initial tensor = {tensor.item()}")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"Process {rank}: After all-reduce tensor: {tensor.item()}")

    dist.destroy_process_group()
    print(f"Process {rank}: Finished")

if __name__ == "__main__":
    world_size = 4

    mp.spawn(
        run_dist_test,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )