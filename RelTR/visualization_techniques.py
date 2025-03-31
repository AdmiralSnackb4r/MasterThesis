

import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.utils as vutils


def plot_heatmap(data, title="Heatmap", cmap="viridis", colorbar=True):
    """
    Plots a heatmap using Matplotlib.

    Args:
        data (numpy.ndarray or torch.Tensor): 2D array representing the heatmap.
        title (str): Title of the heatmap.
        cmap (str): Colormap for visualization.
        colorbar (bool): Whether to show a colorbar.
    """
    # Convert from PyTorch tensor to NumPy if necessary
    if hasattr(data, 'cpu'):  # Check if it's a PyTorch tensor
        data = data.cpu().detach().numpy()

    plt.figure(figsize=(6, 5))  # Set figure size
    plt.imshow(data, cmap=cmap, aspect="auto")  # Plot heatmap
    plt.title(title, fontsize=14)  # Set title
    plt.axis("off")  # Hide axes
    
    if colorbar:
        plt.colorbar()  # Show colorbar

    plt.show()  # Display the heatmap


def plot_backbone(conv_features: torch.tensor, input_image: Image):

    feature_maps = conv_features.squeeze(0) # Shape now [2048, 29, 25]

    width, height = input_image.size
    h, w = feature_maps.shape[-2:]


    summed_featured_map = 0

    scale_factor = int(width/conv_features.shape[-1])
    upsampler = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    feature_maps = feature_maps[-1, :, :]

    for feature_map in feature_maps:
        #summed_featured_map += upsampler(feature_map.view(1, 1, h, w))
        summed_featured_map += F.interpolate(feature_map.view(1, 1, h, w), size=(890, 1178), mode='bilinear', align_corners=True)

    #summed_featured_map = (summed_featured_map - summed_featured_map.min()) / (summed_featured_map.max() - summed_featured_map.min())
    #summed_featured_map = (summed_featured_map + 255).byte()

    summed_featured_map = summed_featured_map.squeeze(0).squeeze(0)
    plt.imshow(summed_featured_map.numpy(), cmap='viridis')  # Apply 'jet' colormap (or others like 'viridis', 'plasma')
    plt.colorbar()  # Optionally add a colorbar for reference

    # Save the image
    plt.axis('off')  # Turn off axes for clean image
    plt.savefig("heatmap_vg1.png", bbox_inches='tight', pad_inches=0)
    plt.close()


    #plot_heatmap(summed_featured_map.squeeze(0).squeeze(0))

    


def attention_rollout(attentions: torch.tensor):
    # Init rollout with identity
    rollout = torch.eye(attentions[0].size(-1)).to(attentions[0].device)
    print(attentions[0].shape)
    print(attentions[1].shape)

    for attention in attentions:
        #print(attention.shape)
        attention_heads_fused = attention.mean(dim=1) # Average attention across heads
        #print(attention_heads_fused.shape)
        attention_heads_fused += torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused.device) # A + I
        attention_heads_fused /= attention_heads_fused.sum(dim=1, keepdim=True) # Normalize
        rollout = torch.matmul(rollout, attention_heads_fused)
    
    print(rollout.shape)

    return rollout