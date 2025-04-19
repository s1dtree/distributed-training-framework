import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time

from cifar10_data_loader import load_cifar10_data
from simple_cnn import SimpleCNN

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch

    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer for updating weights
        device: Device to run training on (cpu or cuda)

    Returns:
        average_loss: Average loss over epoch
        accuracy: Training accuracy
    """

    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for inputs, targets in tqdm(train_loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass & optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += targets.eq(predicted).sum().item()

        average_loss = running_loss / len(train_loader)
        accuracy = 100. * correct / total
        
    return average_loss, accuracy
    
def validate(model, test_loader, criterion, device):
    """
    Validate model on test set

    Args:
        model: Neural network model
        test_loader: DataLoder for test data
        criterion: Loss function
        device: Device to run validation on (cpu or cuda)

    Returns:
        average_loss: Average loss over test set
        accuracy: Test accuracy
    """

    model.eval()
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Validation"):
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

def main():
    parser = argparse.ArgumentParser(description="Train a CNN on CIFAR-10")
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--save-model', action='store_true', default=False, help='Save the trained model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, classes = load_cifar10_data(
        batch_size=args.batch_size,
        num_workers=4 if use_cuda else 2
    )

    model = SimpleCNN().to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.2f}%")

        test_loss, test_acc = validate(model, test_loader, criterion, device)
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.2f}%")

        scheduler.step()

        if test_acc > best_acc and args.save_model:
            best_acc = test_acc
            torch.save(model.state_dict(), "cifar10_cnn_best.pth")
            print(f"Saved model with accuracy: {best_acc:.2f}%")

        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/60:.2f} minutes")
        print(f"Best test accuracy: {best_acc:.2f}%")

        if args.save_model:
            torch.save(model.state_dict(), "cifar10_cnn_final.pth")
            print("Final model saved")

if __name__ == "__main__":
    main()