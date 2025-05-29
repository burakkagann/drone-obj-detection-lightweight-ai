import os
import matplotlib.pyplot as plt

def save_histogram(img_orig, img_aug, filename, augmentation_type, output_dir="checkpoints/augmentation_histograms"):
    os.makedirs(output_dir, exist_ok=True)
    colors = ['b', 'g', 'r']
    plt.figure(figsize=(10, 6))
    for i, color in enumerate(colors):
        plt.hist(img_orig[..., i].ravel(), bins=256, alpha=0.5, label=f'{color.upper()} - Original', color=color)
        plt.hist(img_aug[..., i].ravel(), bins=256, alpha=0.5, linestyle='dashed', label=f'{color.upper()} - {augmentation_type}', edgecolor='black')
    plt.title(f"Histogram - {augmentation_type}: {filename}")
    plt.legend()
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    save_path = os.path.join(output_dir, f"{augmentation_type}_{filename.replace('.jpg', '')}.png")
    plt.savefig(save_path)
    plt.close()
