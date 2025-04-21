import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import time

from cifar10_data_loader import load_cifar10_data
from simple_cnn import SimpleCNN

def train_one_epoch(model, train_loader, criterion, optimizer, device, rank):
    """
    Train model for one epoch

    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for updating weights
        device: Device to run training on (cpu or cuda)
        rank: Process rank

    Returns:
        average_loss: Average loss over epoch
        accuracy: Training accuracy
    """

    model.train()
    running_loss = 0
    correct = 0
    total = 0

    train_iter = tqdm(train_loader, desc="Training") if rank == 0 else train_loader

    for inputs, targets in train_iter:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += targets.eq(predicted).sum().item()

        average_loss = running_loss / len(train_loader)
        accuracy = 100. * correct / total
        
    return average_loss, accuracy
    
def validate(model, test_loader, criterion, device, rank):
    """
    Validate model on test set

    Args:
        model: Neural network model
        test_loader: DataLoder for test data
        criterion: Loss function
        device: Device to run validation on (cpu or cuda)
        rank: Process rank

    Returns:
        average_loss: Average loss over test set
        accuracy: Test accuracy
    """

    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        test_iter = tqdm(test_loader, desc="Validation") if rank == 0 else test_loader
        for inputs, targets in test_iter:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    average_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total

    return average_loss, accuracy   

def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10")
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--save-model', action='store_true', default=False, help='Save the trained model')
    
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', -1)), 
                        help='Local rank for distributed training (-1 for non-distributed)')
    parser.add_argument('--backend', type=str, default='auto', 
                        choices=['auto', 'nccl', 'gloo'],
                        help='Distributed backend: nccl, gloo, or auto (default: auto)')
    
    return parser.parse_args()

def main(args=None, provided_rank=None, provided_world_size=None):
    """
    Main training function, modified to work with simulation
    
    Args:
        args: Command line arguments (if None, will parse from command line)
        provided_rank: Process rank (set by simulation)
        provided_world_size: Total number of processes (set by simulation)
    """
    if args is None:
        args = parse_args()
    
    if provided_rank is not None and provided_world_size is not None:
        rank = provided_rank
        world_size = provided_world_size
        dist_initialized_externally = True
    else:
        dist_initialized_externally = False
        
        if args.local_rank != -1:
            if args.backend == 'auto':
                backend = 'nccl' if torch.cuda.is_available() else 'gloo'
            else:
                backend = args.backend

            dist.init_process_group(backend=backend)
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

    if rank == 0:
        print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        print(f"World size: {world_size}")

    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        if args.local_rank != -1:
            torch.cuda.set_device(args.local_rank)
            device = torch.device(f"cuda:{args.local_rank}")
        else:
            device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    train_dataset, test_dataset, classes = load_cifar10_data(
        batch_size=args.batch_size,
        num_workers=4,
        return_datasets=True
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )

    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    model = SimpleCNN().to(device)

    if world_size > 1:
        if torch.cuda.is_available():
            model = DDP(model, device_ids=[args.local_rank])
        else:
            model = DDP(model)

    if rank == 0:
        print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, rank)
        if rank == 0:   
            print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.2f}%")

        test_loss, test_acc = validate(model, test_loader, criterion, device, rank)
        if rank == 0:
            print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.2f}%")

            if test_acc > best_acc and args.save_model:
                best_acc = test_acc
                torch.save(model.module.state_dict() if world_size > 1 else model.state_dict(), "cifar10_cnn_best.pth")
                print(f"Saved model with accuracy: {best_acc:.2f}%")

        scheduler.step()

    if rank == 0:
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        print(f"Best test accuracy: {best_acc:.2f}%")

        if args.save_model:
            torch.save(model.module.state_dict() if world_size > 1 else model.state_dict(), "cifar10_cnn_final.pth")
            print("Final model saved")
    
    if args.local_rank != -1 and not dist_initialized_externally:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()