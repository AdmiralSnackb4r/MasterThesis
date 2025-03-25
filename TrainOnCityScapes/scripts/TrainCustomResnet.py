import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from CustomCocoDataset import Preparator, CustomCocoDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training will happen on {device}")


# Hyperparams
lr = 0.001
image_size = (224, 224)
batch_size = 8
num_epochs = 10
continue_training = False
train_loop = 2


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize the image
                         std=[0.229, 0.224, 0.225])
])

split_value = 0.8
print(f"Creating Train and Test Datasets with split value of {split_value}")
#preparator = Preparator(None, "CityScapes/coco_annotations.json")
#draws_train, draws_test = preparator.split_train_test(split=split_value)
#preparator.create_split_annotations(draws_train, "train_dataset.json")
#preparator.create_split_annotations(draws_test, "test_dataset.json")

trainset = CustomCocoDataset(root_dir="S:\\Datasets\\CityScapes\\leftImg8bit",
                            annotation_file="train_dataset.json",
                            mode="object",
                            transforms=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle=True)

testset = CustomCocoDataset(root_dir="S:\\Datasets\\CityScapes\\leftImg8bit",
                            annotation_file="test_dataset.json",
                            mode="object",
                            transforms=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle=True)


print("Initialize ResNet")
resnet = models.resnet101(num_classes=27)
resnet.fc = nn.Linear(resnet.fc.in_features, 27)
resnet = resnet.to(device)

# Load pre-trained weights if available
if continue_training:
    pretrained_weights_path = 'training_2/resnet_50_epoch_9_loss_0.6167824279478857.pth'  # Replace X and Y with epoch and loss from your saved model
    if os.path.exists(pretrained_weights_path):
        print(f"Loading pre-trained weights from {pretrained_weights_path}")
        resnet.load_state_dict(torch.load(pretrained_weights_path))
    else:
        print("No pre-trained weights found. Training from scratch.")


criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(resnet.parameters(), lr=0.001)
writer = SummaryWriter('logs/resnet_training')

last_epoch_train_loss = float(np.inf)
last_epoch_test_loss = float(np.inf)


# Training loop
epochs = num_epochs
print("Training Started")
print("Hyperparameters:")
print(f"Learning Rate: {lr}")
print(f"Image Size: {image_size}")
print(f"Batch Size: {batch_size}")
print(f"Number of Epochs: {num_epochs}")

for epoch in range(epochs):
    running_train_loss = 0.0
    running_test_loss = 0.0
    train_flag = False
    test_flag = False
    resnet.train()  # Set model to training mode
    for i_train, sample in enumerate(trainloader, 0):
        inputs, labels = sample['image'], sample['label']
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimize the parameters

        # Print statistics
        running_train_loss += loss.item()
        writer.add_scalar('Running Train Loss', running_train_loss / (i_train + 1),
                           epoch * len(trainloader) + i_train)
        writer.add_scalar('Training Loss', loss.item(), epoch * len(trainloader) + i_train)

    for i_test, test_sample in enumerate(testloader, 0):
        test_inputs, test_labels = test_sample['image'], test_sample['label']
        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
        test_outputs = resnet(test_inputs)
        test_loss = criterion(test_outputs, test_labels)
        running_test_loss += test_loss.item()
        writer.add_scalar(f'Running Test Loss {train_loop}', running_test_loss / (i_test + 1),
                           epoch * len(trainloader) + i_test)
        writer.add_scalar(f'Test Loss {train_loop}', test_loss.item(), epoch * len(testloader) + i_test)
    
    if running_train_loss / (i_train + 1) < last_epoch_train_loss:
        train_flag = True
        last_epoch_train_loss = running_train_loss / (i_train + 1)
    if running_test_loss / (i_test + 1) < last_epoch_test_loss:
        test_flag = True
        last_epoch_test_loss = running_test_loss / (i_test + 1)

    if train_flag and test_flag:
        torch.save(resnet.state_dict(), f'resnet_101_epoch_{epoch}_loss_{last_epoch_test_loss}.pth')


print('Finished Training')

# Save the trained model
torch.save(resnet.state_dict(), 'resnet_cifar10.pth')
