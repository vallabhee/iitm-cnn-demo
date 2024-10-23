import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
from torchvision import transforms as T


def plot_loss(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history["val_loss"], label="val", marker="o")
    plt.plot(history["train_loss"], label="train", marker="o")
    plt.title("Loss per epoch")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(), plt.grid()
    plt.show()


def plot_score(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_miou"], label="train_mIoU", marker="*")
    plt.plot(history["val_miou"], label="val_mIoU", marker="*")
    plt.title("Score per epoch")
    plt.ylabel("mean IoU")
    plt.xlabel("epoch")
    plt.legend(), plt.grid()
    plt.show()


def plot_acc(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_acc"], label="train_accuracy", marker="*")
    plt.plot(history["val_acc"], label="val_accuracy", marker="*")
    plt.title("Accuracy per epoch")
    plt.ylabel("Accuracy")
    plt.xlabel("epoch")
    plt.legend(), plt.grid()
    plt.show()


def visualize_predictions(
    model,
    test_set,
    output_pdf,
    num_classes=23,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
):
    model.eval()
    device = next(model.parameters()).device

    # Create a colormap for the segmentation mask
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in np.linspace(0, 1, num_classes)]

    with PdfPages(output_pdf) as pdf:
        for i, (img, mask) in enumerate(test_set):
            # Prepare the image
            img_tensor = T.Compose([T.ToTensor(), T.Normalize(mean, std)])(img)
            img_tensor = img_tensor.unsqueeze(0).to(device)

            # Get the prediction
            with torch.no_grad():
                output = model(img_tensor)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            # Create a figure with three subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f"Test Image {i+1}")

            # Plot original image
            ax1.imshow(img)
            ax1.set_title("Original Image")
            ax1.axis("off")

            # Plot ground truth mask
            ax2.imshow(mask, cmap=cmap, vmin=0, vmax=num_classes - 1)
            ax2.set_title("Ground Truth")
            ax2.axis("off")

            # Plot predicted mask
            ax3.imshow(pred_mask, cmap=cmap, vmin=0, vmax=num_classes - 1)
            ax3.set_title("Prediction")
            ax3.axis("off")

            # Add the plot to the PDF
            pdf.savefig(fig)
            plt.close(fig)

        print(f"Visualizations saved to {output_pdf}")
