import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from tqdm import tqdm
import time
import numpy as np
from metrics import mIoU, pixel_accuracy
import os
from torchvision import transforms  # Change this line

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model():
    return smp.Unet(
        "mobilenet_v2",
        encoder_weights="imagenet",
        classes=23,
        activation=None,
        encoder_depth=5,
        decoder_channels=[256, 128, 64, 32, 16],
    )


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def fit(
    epochs,
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    patch=False,
    checkpoint_dir="checkpoints",
):
    torch.cuda.empty_cache()
    train_losses = []
    test_losses = []
    val_iou = []
    val_acc = []
    train_iou = []
    train_acc = []
    lrs = []
    min_loss = np.inf
    decrease = 1
    not_improve = 0

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.to(device)
    fit_time = time.time()
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        iou_score = 0
        accuracy = 0
        # training loop
        model.train()
        for i, data in enumerate(tqdm(train_loader)):
            # training phase
            image_tiles, mask_tiles = data
            if patch:
                bs, n_tiles, c, h, w = image_tiles.size()
                image_tiles = image_tiles.view(-1, c, h, w)
                mask_tiles = mask_tiles.view(-1, h, w)

            image = image_tiles.to(device)
            mask = mask_tiles.to(device)
            # forward
            output = model(image)
            loss = criterion(output, mask)
            # evaluation metrics
            iou_score += mIoU(output, mask)
            accuracy += pixel_accuracy(output, mask)
            # backward
            loss.backward()
            optimizer.step()  # update weight
            optimizer.zero_grad()  # reset gradient

            # step the learning rate
            lrs.append(get_lr(optimizer))
            scheduler.step()

            running_loss += loss.item()

        else:
            model.eval()
            test_loss = 0
            test_accuracy = 0
            val_iou_score = 0
            # validation loop
            with torch.no_grad():
                for i, data in enumerate(tqdm(val_loader)):
                    # reshape to 9 patches from single image, delete batch size
                    image_tiles, mask_tiles = data

                    if patch:
                        bs, n_tiles, c, h, w = image_tiles.size()
                        image_tiles = image_tiles.view(-1, c, h, w)
                        mask_tiles = mask_tiles.view(-1, h, w)

                    image = image_tiles.to(device)
                    mask = mask_tiles.to(device)
                    output = model(image)
                    # evaluation metrics
                    val_iou_score += mIoU(output, mask)
                    test_accuracy += pixel_accuracy(output, mask)
                    # loss
                    loss = criterion(output, mask)
                    test_loss += loss.item()

            # calculation mean for each batch
            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(val_loader))

            if min_loss > (test_loss / len(val_loader)):
                print(
                    "Loss Decreasing.. {:.3f} >> {:.3f} ".format(
                        min_loss, (test_loss / len(val_loader))
                    )
                )
                min_loss = test_loss / len(val_loader)
                decrease += 1
                if decrease % 5 == 0:
                    print("saving model...")
                    torch.save(
                        model,
                        "Unet-Mobilenet_v2_mIoU-{:.3f}.pt".format(
                            val_iou_score / len(val_loader)
                        ),
                    )

            if (test_loss / len(val_loader)) > min_loss:
                not_improve += 1
                min_loss = test_loss / len(val_loader)
                print(f"Loss Not Decrease for {not_improve} time")
                if not_improve == 7:
                    print("Loss not decrease for 7 times, Stop Training")
                    break

            # iou
            val_iou.append(val_iou_score / len(val_loader))
            train_iou.append(iou_score / len(train_loader))
            train_acc.append(accuracy / len(train_loader))
            val_acc.append(test_accuracy / len(val_loader))
            print(
                "Epoch:{}/{}..".format(e + 1, epochs),
                "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
                "Val Loss: {:.3f}..".format(test_loss / len(val_loader)),
                "Train mIoU:{:.3f}..".format(iou_score / len(train_loader)),
                "Val mIoU: {:.3f}..".format(val_iou_score / len(val_loader)),
                "Train Acc:{:.3f}..".format(accuracy / len(train_loader)),
                "Val Acc:{:.3f}..".format(test_accuracy / len(val_loader)),
                "Time: {:.2f}m".format((time.time() - since) / 60),
            )

        # Save checkpoint
        history = {
            "train_loss": train_losses,
            "val_loss": test_losses,
            "train_miou": train_iou,
            "val_miou": val_iou,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lrs": lrs,
        }
        save_checkpoint(
            model, optimizer, scheduler, e + 1, min_loss, history, checkpoint_dir
        )

    print("Total time: {:.2f} m".format((time.time() - fit_time) / 60))
    return history


def predict_image_mask_miou(
    model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
):
    model.eval()
    t = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )  # Change this line
    image = t(image)
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        output = model(image)
        score = mIoU(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, score


def predict_image_mask_pixel(
    model, image, mask, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
):
    model.eval()
    t = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )  # Change this line
    image = t(image)
    model.to(device)
    image = image.to(device)
    mask = mask.to(device)
    with torch.no_grad():
        image = image.unsqueeze(0)
        mask = mask.unsqueeze(0)
        output = model(image)
        acc = pixel_accuracy(output, mask)
        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)
    return masked, acc


def evaluate_model(model, test_set):
    score_iou = []
    accuracy = []
    for i in tqdm(range(len(test_set))):
        img, mask = test_set[i]
        pred_mask, score = predict_image_mask_miou(model, img, mask)
        score_iou.append(score)
        _, acc = predict_image_mask_pixel(model, img, mask)
        accuracy.append(acc)
    return np.mean(score_iou), np.mean(accuracy)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    history = {
        "train_loss": checkpoint["train_loss"],
        "val_loss": checkpoint["val_loss"],
        "train_miou": checkpoint["train_miou"],
        "val_miou": checkpoint["val_miou"],
        "train_acc": checkpoint["train_acc"],
        "val_acc": checkpoint["val_acc"],
        "lrs": checkpoint["lrs"],
    }
    return model, optimizer, scheduler, epoch, loss, history


def get_latest_checkpoint(checkpoint_dir):
    checkpoints = [
        f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_epoch_")
    ]
    if not checkpoints:
        return None
    latest_checkpoint = max(
        checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    return os.path.join(checkpoint_dir, latest_checkpoint)


def resume_from_checkpoint(model, optimizer, scheduler, checkpoint_dir):
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        model, optimizer, scheduler, start_epoch, min_loss, history = load_checkpoint(
            model, optimizer, scheduler, latest_checkpoint
        )
        print(f"Resuming training from epoch {start_epoch}")
        return model, optimizer, scheduler, start_epoch, min_loss, history
    else:
        print("No checkpoint found. Starting from scratch.")
        return model, optimizer, scheduler, 0, float("inf"), {}


def save_checkpoint(model, optimizer, scheduler, epoch, loss, history, checkpoint_dir):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "train_loss": history.get("train_loss", []),
        "val_loss": history.get("val_loss", []),
        "train_miou": history.get("train_miou", []),
        "val_miou": history.get("val_miou", []),
        "train_acc": history.get("train_acc", []),
        "val_acc": history.get("val_acc", []),
        "lrs": history.get("lrs", []),
    }
    torch.save(
        checkpoint, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    )
