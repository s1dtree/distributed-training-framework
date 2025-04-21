import os
import sys
import argparse
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

def setup_process(rank, world_size, args, fn):
    """
    Initialize the distributed environment for a single process

    Args:
        rank (int): Global rank of the process
        world_size (int): Total number of processes
        args (argparse.Namespace): Arguments for the training process
        fn (callable): Function to run after initialization
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(args.port)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)

    args.local_rank = 0

    if args.backend == 'auto':
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    else:
        backend = args.backend

    print(f"Process {rank}: Initializing process group with {backend} backend")

    dist.init_process_group(backend=backend)

    fn(args, rank, world_size)

    dist.destroy_process_group()
    print(f"Process {rank}: Process group destroyed")

def simulate_multi_node(args, fn):
    """
    Launch multiple processes to simulate a multi-node setup.

    Args:
        args (argparse.Namespace): Arguments for the simulation
        fn (callable): Function to run on each simulated node
    """
    world_size = args.num_nodes
    print(f"Simulating {world_size} nodes on a single machine")

    mp.spawn(
        setup_process,
        args=(world_size, args, fn),
        nprocs=world_size,
        join=True
    )

def parse_args():
    """Parse command line arguments for multi-node simulation"""

    parser = argparse.ArgumentParser(description='Multi-node distributed training simulation')

    parser.add_argument('--num_nodes', type=int, default=2, 
                        help='Number of nodes to simulate (default: 2)')
    parser.add_argument('--port', type=int, default=29500, 
                        help='Port used for distributed training (default: 29500)')
    
    parser.add_argument('--backend', type=str, default='auto', 
                        choices=['auto', 'nccl', 'gloo'],
                        help='Distributed backend: nccl, gloo, or auto (default: auto)')
    
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Input batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, 
                        help='Number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, 
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed (default: 42)')
    parser.add_argument('--save_model', action='store_true', default=False, 
                        help='Save the trained model')
    
    parser.add_argument('--local_rank', type=int, default=-1, 
                        help='Local rank for distributed training (-1 for non-distributed)')
    
    return parser.parse_args()

def main_worker(args, rank, world_size):
    """
    The main function that will run on each simulated node
    This contains the actual distributed training code

    Args:
        args (argparse.Namespace): Command line arguments
        rank (int): Global rank of this process
        world_size (int): Total number of processes
    """
    
    from distributed_train import main as distributed_main
    
    distributed_main(args, rank, world_size)

if __name__ == "__main__":
    args = parse_args()
    simulate_multi_node(args, main_worker)