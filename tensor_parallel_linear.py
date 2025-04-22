import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

class ColumnParallelLinear(nn.Module):
    """
    Column-parallel linear layer implementation
    Weight matrix is split along the output dimension (column)
    """
    def __init__(self, input_size, output_size, bias=True, world_size=None, rank=None):
        super(ColumnParallelLinear, self).__init__()
        
        self.world_size = world_size if world_size is not None else dist.get_world_size()
        self.rank = rank if rank is not None else dist.get_rank()
        
        self.output_size_per_partition = output_size // self.world_size
        assert output_size % self.world_size == 0, "Output size must be divisible by world size"
        
        self.linear = nn.Linear(input_size, self.output_size_per_partition, bias=bias)
    
    def forward(self, input_):
        local_output = self.linear(input_)
        
        output_list = [torch.zeros_like(local_output) for _ in range(self.world_size)]
        dist.all_gather(output_list, local_output)
        
        output = torch.cat(output_list, dim=-1)
        
        return output

class RowParallelLinear(nn.Module):
    """
    Row-parallel linear layer implementation
    The weight matrix is split along the input dimension (rows)
    """
    def __init__(self, input_size, output_size, bias=True, world_size=None, rank=None):
        super(RowParallelLinear, self).__init__()
        
        self.world_size = world_size if world_size is not None else dist.get_world_size()
        self.rank = rank if rank is not None else dist.get_rank()
        
        self.input_size_per_partition = input_size // self.world_size
        assert input_size % self.world_size == 0, "Input size must be divisible by world size"
        
        self.linear = nn.Linear(self.input_size_per_partition, output_size, bias=bias if rank == 0 else False)
        
        self.use_bias = bias and rank == 0
    
    def forward(self, input_):
        input_parallel = input_[:, self.rank * self.input_size_per_partition:(self.rank + 1) * self.input_size_per_partition]
        
        local_output = self.linear(input_parallel)
        
        output = local_output.clone()
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        
        return output

def init_weights(module):
    """Initialize weights with known values for testing"""
    if isinstance(module, nn.Linear):
        nn.init.constant_(module.weight, 0.01 * (dist.get_rank() + 1))
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.1 * (dist.get_rank() + 1))

def setup(rank, world_size):
    """Initialize the distributed environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world_size)
    
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

def cleanup():
    """Clean up the distributed environment"""
    dist.destroy_process_group()

def test_column_parallel_linear(rank, world_size):
    """Test column-parallel linear layer"""
    setup(rank, world_size)
    
    input_size = 4
    output_size = 6
    
    model = ColumnParallelLinear(input_size, output_size, bias=True)
    model.apply(init_weights)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    input_tensor = torch.ones(2, input_size, device=device)
    
    output = model(input_tensor)
    
    print(f"Rank {rank}: Input shape = {input_tensor.shape}, Output shape = {output.shape}")
    print(f"Rank {rank}: Local linear weight = {model.linear.weight.data}")
    print(f"Rank {rank}: Output = {output}")
    
    cleanup()

def test_row_parallel_linear(rank, world_size):
    """Test row-parallel linear layer"""
    setup(rank, world_size)
    
    input_size = 8
    output_size = 4
    
    model = RowParallelLinear(input_size, output_size, bias=True)
    model.apply(init_weights)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    input_tensor = torch.ones(2, input_size, device=device)
    
    output = model(input_tensor)
    
    print(f"Rank {rank}: Input shape = {input_tensor.shape}, Output shape = {output.shape}")
    print(f"Rank {rank}: Local linear weight = {model.linear.weight.data}")
    print(f"Rank {rank}: Local linear input = {input_tensor[:, rank * model.input_size_per_partition:(rank + 1) * model.input_size_per_partition]}")
    print(f"Rank {rank}: Output = {output}")
    
    cleanup()

def test_two_layer_mlp(rank, world_size):
    """Test a 2-layer MLP with mixed parallelism"""
    setup(rank, world_size)
    
    input_size = 8
    hidden_size = 12
    output_size = 4
    
    first_layer = ColumnParallelLinear(input_size, hidden_size, bias=True)
    activation = nn.ReLU()
    second_layer = RowParallelLinear(hidden_size, output_size, bias=True)
    
    first_layer.apply(init_weights)
    second_layer.apply(init_weights)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    first_layer = first_layer.to(device)
    second_layer = second_layer.to(device)

    input_tensor = torch.ones(2, input_size, device=device)
    
    hidden = first_layer(input_tensor)
    hidden = activation(hidden)
    output = second_layer(hidden)
    
    print(f"Rank {rank}: Input shape = {input_tensor.shape}")
    print(f"Rank {rank}: Hidden shape = {hidden.shape}")
    print(f"Rank {rank}: Output shape = {output.shape}")
    print(f"Rank {rank}: Output = {output}")
    
    cleanup()

if __name__ == "__main__":
    TEST_TYPE = "two_layer_mlp"
    
    WORLD_SIZE = 2
    
    if TEST_TYPE == "column":
        mp.spawn(test_column_parallel_linear, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
    elif TEST_TYPE == "row":
        mp.spawn(test_row_parallel_linear, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)
    elif TEST_TYPE == "two_layer_mlp":
        mp.spawn(test_two_layer_mlp, args=(WORLD_SIZE,), nprocs=WORLD_SIZE, join=True)