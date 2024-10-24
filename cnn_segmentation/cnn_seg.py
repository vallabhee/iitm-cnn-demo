import torch
import torch.nn as nn
from cnn_data import get_data_loaders
from model import create_model, fit, evaluate_model, resume_from_checkpoint
from viz import plot_loss, plot_score, plot_acc, visualize_predictions
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model():
    # Get data loaders
    train_loader, val_loader, test_set = get_data_loaders(batch_size=16)

    # Create model
    model = create_model()

    # Training parameters
    max_lr = 1e-3
    epochs = 30
    weight_decay = 1e-4

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=max_lr, weight_decay=weight_decay
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader)
    )

    # Check for existing checkpoints and resume training if possible
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model, optimizer, sched, start_epoch, min_loss, history = resume_from_checkpoint(
        model, optimizer, sched, checkpoint_dir
    )

    # Train the model
    history = fit(
        epochs - start_epoch,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        sched,
        checkpoint_dir=checkpoint_dir,
    )

    # Save the final model
    torch.save(model, "Unet-Mobilenet.pt")

    # Plot training results
    plot_loss(history)
    plot_score(history)
    plot_acc(history)

    # Evaluate on test set
    test_miou, test_accuracy = evaluate_model(model, test_set)
    print("Test Set mIoU:", test_miou)
    print("Test Set Pixel Accuracy:", test_accuracy)

    # Visualize predictions
    visualize_predictions(model, test_set, "test_predictions.pdf")

    return model


def inference(model_path):
    # Get data loaders
    _, _, test_set = get_data_loaders(batch_size=16)

    # Load the trained model
    model = torch.load(model_path, map_location=device)
    print(f"Loaded model from {model_path}")

    # Evaluate on test set
    test_miou, test_accuracy = evaluate_model(model, test_set)
    print("Test Set mIoU:", test_miou)
    print("Test Set Pixel Accuracy:", test_accuracy)

    # Visualize predictions
    visualize_predictions(model, test_set, "test_predictions.pdf")


# Example usage in a notebook:
# To train a new model:
# model = train_model()

# To perform inference with a trained model:
# inference("path/to/your/model.pt")
