import numpy as np
import matplotlib.pyplot as plt

import torch


def visualize_predictions(model, val_loader, device, num_samples=3):
    model.eval()
    samples = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds_binary = (preds > 0.5).float()

            for i in range(x.size(0)):
                if samples >= num_samples:
                    break

                plt.figure(figsize=(20, 5))

                plt.subplot(1, 4, 1)
                plt.title("Original image")
                plt.imshow(np.transpose(x[i].cpu().numpy(), (1, 2, 0)))

                plt.subplot(1, 4, 2)
                plt.title("Ground truth mask")
                plt.imshow(y[i].cpu().numpy().squeeze(), cmap="gray")

                plt.subplot(1, 4, 3)
                plt.title("Predicted segmentation mask")
                plt.imshow(preds_binary[i].cpu().numpy().squeeze(), cmap="gray")

                plt.subplot(1, 4, 4)
                plt.title("Original + predicted mask")
                original_image = np.transpose(x[i].cpu().numpy(), (1, 2, 0))
                plt.imshow(original_image)

                mask = preds_binary[i].cpu().numpy().squeeze()
                mask_border = np.zeros_like(mask)
                mask_border[mask == 1] = 1
                plt.contour(mask_border, colors="yellow", linewidths=2)

                plt.show()
                samples += 1

            if samples >= num_samples:
                break
