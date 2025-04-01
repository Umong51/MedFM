import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from safetensors.torch import load_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import config as cfg
import lidc
import metrics
from model import Sampler, build_model

device = "cuda" if torch.cuda.is_available() else "cpu"

test_dataset = lidc.test_dataset()
test_loader = DataLoader(test_dataset, batch_size=32)

model = build_model(cfg)
load_model(model, "checkpoints/last.safetensors")
model.to(device)

model.eval()

sampler = Sampler(model, num_steps=cfg.num_steps)

n_samples = 16

geds = 0
h_ious = 0
count = 0

pbar = tqdm(test_loader)

for images, labels, _ in pbar:
    images, labels = images.to(device), labels.to(device)
    preds = sampler.sample_n(images, n_samples)

    labels = labels.argmax(dim=2)
    ged, _, _ = metrics.batched_ged(labels, preds, lidc.NUM_CLASSES)
    h_iou = metrics.batched_hungarian_iou(labels, preds, lidc.NUM_CLASSES)

    geds += ged.sum().item()
    h_ious += h_iou.sum().item()
    count += len(images)

    pbar.set_postfix(
        {
            "GED:": geds / count,
            "H-IoU": h_ious / count,
        }
    )

avg_ged = geds / len(test_dataset)
avg_h_iou = h_ious / len(test_dataset)

print("GED:", avg_ged)
print("H-IoU:", avg_h_iou)
