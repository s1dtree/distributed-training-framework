import os
import torch
import torch.distributed as dist

def main():
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')

    tensor = torch.tensor([float(rank + 1)])
    print(f"Process {rank}: Initial tensor = {tensor.item()}")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"Process {rank}: After all-reduce tensor = {tensor.item()}")

    dist.destroy_process_group()
    print(f"Process {rank}: Finished")

if __name__ == "__main__":
    main()