import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from CustomCocoDataset import CustomCocoDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing will happen on {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize the image
                         std=[0.229, 0.224, 0.225])
])

testset = CustomCocoDataset(root_dir="S:\\Datasets\\CityScapes\\leftImg8bit",
                            annotation_file="test_dataset.json",
                            mode="object",
                            transforms=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size = 1, shuffle=True)

print("Load ResNet")
resnet = models.resnet50(weights=None, num_classes=27)
resnet.fc = nn.Linear(resnet.fc.in_features, 27)
reset = resnet.to(device)

print("Load trained weights")
pretrained_weights_path = 'resnet_cifar10.pth'
reset.load_state_dict(torch.load(pretrained_weights_path))
resnet.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for i_test, sample in enumerate(testloader, 0):
        images, labels = sample['image'], sample['label']
        images, labels = images.to(device), labels.to(device)
        outputs = reset(images)
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics here
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1 = f1_score(all_labels, all_predictions, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
