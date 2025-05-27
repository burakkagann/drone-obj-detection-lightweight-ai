import matplotlib.pyplot as plt

def plot_histograms(img_orig, img_aug):
    """
    Compare histograms of original and augmented images.
    """
    colors = ['b', 'g', 'r']
    for i, color in enumerate(colors):
        plt.hist(img_orig[..., i].ravel(), bins=256, alpha=0.5, label=f'{color.upper()} - original', color=color)
        plt.hist(img_aug[..., i].ravel(), bins=256, alpha=0.5, linestyle='dashed', label=f'{color.upper()} - augmented', edgecolor='black')
    plt.title("Histogram Comparison")
    plt.legend()
    plt.xlabel("Pixel value")
    plt.ylabel("Frequency")
    plt.show()
