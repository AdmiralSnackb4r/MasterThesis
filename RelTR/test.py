import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
 
# Bild laden und vorbereiten
def load_image(img_path, max_size=400):
    image = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(max_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :]),  # RGB
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    return image
 
content = load_image("S:\\Datasets\\CityScapes\\leftImg8bit\\train_extra\\bayreuth\\bayreuth_000000_000003_leftImg8bit.png").to("cuda")
style = load_image("F:\\scenario_runner-0.9.15\\Data\\_out\\OppositeVehicleRunningRedLight_5_3\\rgb\\filtered\\00002064.png").to("cuda")
 
# VGG19 laden
vgg = models.vgg19(pretrained=True).features.to("cuda").eval()
 
# Zielbild (zu optimieren)
target = nn.Parameter(content.clone().detach())
 
# Feature-Layers definieren
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
 
# Feature Extraction
def get_features(x, model, layers):
    features = {}
    i = 0
    for layer in model.children():
        x = layer(x)
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
            if name in layers:
                features[name] = x

    return features
 
# Gram-Matrix
def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    tensor = tensor.reshape(c, h * w)
    return torch.mm(tensor, tensor.t()) / (c * h * w)
 
# Features berechnen
content_features = get_features(content, vgg, content_layers)
content_features = {k: v.detach() for k, v in content_features.items()}
style_features = get_features(style, vgg, style_layers)
style_grams = {layer: gram_matrix(style_features[layer].detach()) for layer in style_features}
 
# Optimierer
optimizer = optim.Adam([target], lr=0.003)
 
# Loss-Funktionen
style_weight = 1e6
content_weight = 1
 
# Optimierung
for step in range(800):
    target_features = get_features(target, vgg, content_layers + style_layers)
 
    content_loss = torch.mean((target_features['conv_4'] - content_features['conv_4'])**2)
    style_loss = 0
    for layer in style_layers:
        target_gram = gram_matrix(target_features[layer])
        style_gram = style_grams[layer]
        style_loss += torch.mean((target_gram - style_gram)**2)
    total_loss = content_weight * content_loss + style_weight * style_loss
 
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
 
# Ergebnis anzeigen
def im_convert(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0)  # (C, H, W) → (H, W, C)
    return image
 
plt.imshow(im_convert(target))
plt.axis("off")
plt.show()