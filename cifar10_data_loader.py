import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

def load_cifar10_data(batch_size=64, num_workers=2):
    """
    Load and preprocess CIFAR-10 dataset

    Args:
        batch_size (int): # of samples per batch
        num_workers (int): # of subprocesses for data loading

    Returns:
        train_loader (DataLoader): DataLoader for the training set
        test_loader (DataLoader): DataLoader for the test set
        classes (tuple): Class names for CIFAR-10
    """

    # Define data transformation for training data
    # Transformations applied to each image
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2470, 0.2435, 0.2616))
    ])

    # Define data transformations for test data
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                             (0.2470, 0.2435, 0.2616))
    ])

    # Download and load training data
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    # Create DataLoader for training data
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    # Download and load test data
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )

    # Create DataLoader for test data
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Class names for CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes

# Helper function to show images from dataset
def show_images(dataloader, classes):
    """
    Display images from dataloader with labels

    Args:
        dataloader (DataLoader): DataLoader containing images to display
        classes (tuple): Class names for labeling
    """

    # Get batch of images
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    # Convert tensors to numpy for display
    images = images.cpu().numpy()

    # Create grid of images
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()

    # Display images
    for i in range(10):
        # Transform & denormalize image
        img = np.transpose(images[i], (1, 2, 0))
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
        img = img * std + mean
        img = np.clip(img, 0, 1)

        # Display image & label
        axes[i].imshow(img)
        axes[i].set_title(classes[labels[i]])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load CIFAR-10 data
    train_loader, test_loader, classes = load_cifar10_data(batch_size=64)
    
    # Display sample images
    print("Displaying sample images from the training set:")
    show_images(train_loader, classes)
    
    # Print dataset information
    print(f"Training dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    print(f"Number of batches in training: {len(train_loader)}")
    print(f"Number of batches in testing: {len(test_loader)}")
    print(f"Number of classes: {len(classes)}")
    
    # Extract one batch to examine its shape
    for images, labels in train_loader:
        print(f"Batch image shape: {images.shape}")
        print(f"Batch label shape: {labels.shape}")
        break