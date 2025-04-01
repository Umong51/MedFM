import copy
from pathlib import Path

import torch
import torch.optim as optim
from safetensors.torch import save_model
from torch.utils.data import DataLoader
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from tqdm.auto import tqdm

import config as cfg
import lidc
import metrics
from model import Sampler, build_model, ema

device = "cuda" if torch.cuda.is_available() else "cpu"

train_dataset = lidc.training_dataset()
val_dataset = lidc.validation_dataset(max_size=cfg.val_max_size)

train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size * 2)

model = build_model(cfg)
model.to(device)

# EMA
ema_model = copy.deepcopy(model)

# Show model size
model_size = 0
for param in model.parameters():
    model_size += param.data.nelement()
print("Model params: %.2f M" % (model_size / 1024 / 1024))

optimizer = optim.Adam(model.parameters(), cfg.learning_rate)

total_iters = cfg.max_epochs * len(train_loader)
scheduler = optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=1.0,
    end_factor=cfg.final_lr / cfg.learning_rate,
    total_iters=total_iters,
)

FM = ConditionalFlowMatcher(sigma=0.0)

ckpt_dir = Path(cfg.ckpt_dir)
ckpt_dir.mkdir(exist_ok=True)

pbar = tqdm(total=total_iters)

for epoch in range(cfg.max_epochs):
    model.train()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # [0, 1] -> [-1, 1]
        x1 = 2 * labels[:, [1]] - 1
        x0 = torch.randn_like(x1)

        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

        xt_and_cond = torch.cat([xt, images], dim=1)
        vt = model(t, xt_and_cond)
        loss = torch.mean((vt - ut) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()
        final_lr = scheduler.get_last_lr()[0]

        ema(model, ema_model, cfg.ema_alpha)

        pbar.set_postfix(
            {
                "train/loss": loss.item(),
                "epoch": epoch,
                "lr": final_lr,
            }
        )
        pbar.update()

    save_model(ema_model, ckpt_dir / "last.safetensors")

    if (epoch + 1) % cfg.eval_freq != 0:
        continue

    # Evaluation

    ema_model.eval()

    sampler = Sampler(ema_model, num_steps=cfg.num_steps)

    geds = 0
    h_ious = 0
    count = 0

    pbar_val = tqdm(val_loader)

    for images, labels, _ in pbar_val:
        images, labels = images.to(device), labels.to(device)
        preds = sampler.sample_n(images, cfg.samples)

        labels = labels.argmax(dim=2)
        ged, _, _ = metrics.batched_ged(labels, preds, lidc.NUM_CLASSES)
        h_iou = metrics.batched_hungarian_iou(labels, preds, lidc.NUM_CLASSES)

        geds += ged.sum().item()
        h_ious += h_iou.sum().item()
        count += len(images)

        pbar_val.set_postfix(
            {
                "GED:": geds / count,
                "H-IoU": h_ious / count,
            }
        )

    avg_ged = geds / len(val_dataset)
    avg_h_iou = h_ious / len(val_dataset)

    print("GED:", avg_ged)
    print("H-IoU:", avg_h_iou)

    ckpt_name = f"checkpoint-{epoch + 1}-ged-{avg_ged:.3f}.safetensors"
    save_model(ema_model, ckpt_dir / ckpt_name)
