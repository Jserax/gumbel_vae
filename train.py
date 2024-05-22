import glob

import torch
from PIL import Image
from torch import FloatTensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.notebook import tqdm
from util import LRCosineScheduler, cos_anneal
from vqvae import VQVae


class VAEDataset(Dataset):
    def __init__(
        self,
        img_dir: str,
    ) -> None:
        super().__init__()
        self.imgs = glob.glob(f"{img_dir}/*.jpg")
        self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx) -> FloatTensor:
        image = Image.open(self.imgs[idx]).convert("RGB")
        image = self.transform(image)
        return image


epochs = 16
warmup_epochs = 4
decay_epochs = 2
batch_size = 128
start_lr = 3e-5
max_lr = 6e-4
min_lr = 1e-7
weight_decay = 0.03
grad_clip = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = VAEDataset("/content/dataset")
train_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=2)
epoch_iters = len(train_loader)

model = VQVae(3, 64, 2048, 64).to(device)
optimizer = torch.optim.AdamW(model.parameters())
scaler = torch.cuda.amp.GradScaler()
scheduler = LRCosineScheduler(
    optimizer,
    warmup_epochs * epoch_iters,
    (epochs - decay_epochs) * epoch_iters,
    start_lr,
    min_lr,
    max_lr,
)

temp_decay_iters = 12 * epoch_iters
temp_from = 1.0
temp_to = 1.0 / 16
kld_warmup_iters = 4 * epoch_iters
kld_from = 0.0
kld_to = 6e-3

for epoch in tqdm(range(1, epochs + 1)):
    train_loss = 0.0
    for iteration, image in enumerate(train_loader, start=1):
        image = image.to(device)
        scheduler.step()
        model.vector_quantizer.temperature = cos_anneal(
            0, temp_decay_iters, temp_from, temp_to, iteration
        )
        model.vector_quantizer.kld_scale = cos_anneal(
            0, kld_warmup_iters, kld_from, kld_to, iteration
        )
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.float16):
            loss = model(image)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
        if iteration % (epoch_iters // 10) == 0:
            print(
                f"Epochs: {epoch}/{epochs} | Iters: {iteration}/{epoch_iters} | Train_loss {train_loss / iteration:.4f} | Last loss {loss.item():.4f}"
            )
    torch.save(model.vector_quantizer.state_dict(), "vqvae.pt")
    torch.save(model.decoder.state_dict(), "vq_decoder.pt")
    torch.save(model.encoder.state_dict(), "vq_encoder.pt")
