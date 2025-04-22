

import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision.transforms.functional as TF



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

def visualize_attention_map(image_tensor, attn_weights, query_idx=0, head_idx=0):
    """
    image_tensor: [3, H, W] – the original image fed into the model
    attn_weights: [num_queries, num_heads, num_keys] or [batch, ...]
    """
    # Assume [num_queries, num_heads, num_keys]
    if attn_weights.dim() == 4:
        attn_weights = attn_weights[0]  # drop batch

    num_queries, num_heads, num_keys = attn_weights.shape
    h, w = image_tensor.shape[1:]

    # DETR usually uses 32x downsampling: divide spatial dims by 32
    grid_size = (h // 32, w // 32)

    weights = attn_weights[query_idx, head_idx]  # [num_keys]
    weights = weights.view(grid_size).detach().cpu().numpy()

    # Normalize and upscale
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
    weights = torch.tensor(weights).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    weights = torch.nn.functional.interpolate(weights, size=(h, w), mode='bilinear', align_corners=False)[0,0]

    # Plot
    fig, ax = plt.subplots()
    img = TF.to_pil_image(image_tensor.cpu())
    ax.imshow(img)
    ax.imshow(weights, cmap='jet', alpha=0.5)
    ax.set_title(f"Attention Map: Query {query_idx}, Head {head_idx}")
    ax.axis('off')
    plt.show()

def visualize_multiple_attention_maps(image, attn_weights, queries=(0, 1, 2), heads=(0, 1), title_prefix="", image_size=(0, 0)):
    """
    image_tensor: [3, H, W]
    attn_weights: [num_queries, num_heads, num_keys] or [batch_size, num_queries, num_heads, num_keys]
    queries: tuple of query indices to visualize
    heads: tuple of head indices to visualize
    """
    if attn_weights.dim() == 4:
        attn_weights = attn_weights[0]  # Remove batch dimension

    print(attn_weights.shape)

    num_queries, num_heads, num_keys = attn_weights.shape
    h, w = image_size
    
    num_keys = attn_weights.shape[-1]
    feat_h = int(h / 32)
    feat_w = int(w / 32)

    # Sometimes feature maps are not exactly h/32 x w/32
    # So fall back to sqrt if needed
    if feat_h * feat_w != num_keys:
        sqrt_keys = int(num_keys**0.5)
        if sqrt_keys * sqrt_keys == num_keys:
            feat_h = feat_w = sqrt_keys
        else:
            raise ValueError(f"Can't infer grid size from {num_keys} tokens")

    grid_size = (feat_h, feat_w)

    fig, axes = plt.subplots(len(queries), len(heads), figsize=(4 * len(heads), 4 * len(queries)))
    if len(queries) == 1:
        axes = [axes]
    if len(heads) == 1:
        axes = [[ax] for ax in axes]

    #img = TF.to_pil_image(image_tensor.cpu())

    for i, q_idx in enumerate(queries):
        for j, h_idx in enumerate(heads):
            ax = axes[i][j]
            weights = attn_weights[q_idx, h_idx]  # [num_keys]
            weights = weights.view(grid_size).detach().cpu()
            weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)

            # Upscale to match image size
            weights = torch.nn.functional.interpolate(
                weights.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False
            )[0, 0]

            ax.imshow(image)
            ax.imshow(weights, cmap='jet', alpha=0.5)
            ax.set_title(f"{title_prefix}Q{q_idx} H{h_idx}")
            ax.axis('off')

    plt.tight_layout()
    plt.show()