import torch, gc
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_scheduler
from pathlib import Path
import torchvision.transforms as T

OUTPUT_DIR = Path("/kaggle/working/trocr-finetuned")
MODEL_NAME = "microsoft/trocr-base-printed"
BATCH_SIZE = 32
EPOCHS = 5              # Меньше эпох!
LR = 2e-5               # Меньше LR!

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device: {device}")

gc.collect()
torch.cuda.empty_cache()

# Data augmentation (регуляризация!)
train_transform = T.Compose([
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.RandomAffine(degrees=2, translate=(0.05, 0.05)),
])

class OcrData(Dataset):
    def __init__(self, items, processor, max_len=128, augment=False):
        self.items = items
        self.processor = processor
        self.max_len = max_len
        self.augment = augment
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        item = self.items[idx]
        try:
            img = Image.open(item["image_path"]).convert("RGB")
            if self.augment:
                img = train_transform(img)  # Аугментация!
        except:
            img = Image.new("RGB", (384, 64), 255)
        pix = self.processor(img, return_tensors="pt").pixel_values[0]
        lbl = self.processor.tokenizer(
            item['text'], padding='max_length', max_length=self.max_len,
            truncation=True, return_tensors='pt',
        ).input_ids[0]
        return {'pixel_values': pix, 'labels': lbl}

def collate(batch):
    pix = torch.stack([b['pixel_values'] for b in batch])
    lbl = torch.stack([b['labels'] for b in batch])
    lbl[lbl == processor.tokenizer.pad_token_id] = -100
    return {'pixel_values': pix.to(device), 'labels': lbl.to(device)}

processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).to(device)

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
model.generation_config.pad_token_id = model.config.pad_token_id

# Freeze encoder
for p in model.encoder.parameters():
    p.requires_grad = False

LIMIT = 10000
train_subset = train_data[:LIMIT]
val_subset = val_data[:LIMIT//10]

# Аугментация только для train
train_loader = DataLoader(
    OcrData(train_subset, processor, augment=True), batch_size=BATCH_SIZE,
    shuffle=True, collate_fn=collate, num_workers=0
)
val_loader = DataLoader(
    OcrData(val_subset, processor, augment=False), batch_size=BATCH_SIZE,
    shuffle=False, collate_fn=collate, num_workers=0
)

opt = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR, weight_decay=0.01
)

# Cosine scheduler with warmup
total_steps = EPOCHS * len(train_loader)
sched = get_scheduler('cosine', opt, int(0.1 * total_steps), total_steps)

logger.info(f"Training: {EPOCHS} epochs, {total_steps} steps (lr={LR})")

best_val = float('inf')
OUTPUT_DIR.mkdir(exist_ok=True)

for epoch in range(EPOCHS):
    model.train()
    tloss = 0.0
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
    for batch in pbar:
        loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        tloss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    logger.info(f'Train loss: {tloss/len(train_loader):.4f}')

    model.eval()
    vloss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Val'):
            loss = model(**batch).loss
            vloss += loss.item()
    avg_vloss = vloss / len(val_loader)
    logger.info(f'Val loss: {avg_vloss:.4f}')
    
    # Ранняя остановка при переобучении
    if avg_vloss > best_val and epoch > 1:
        logger.warning(f"Val loss вырос! Остановка.")
        break

    if avg_vloss < best_val:
        best_val = avg_vloss
        model.save_pretrained(str(OUTPUT_DIR))
        processor.save_pretrained(str(OUTPUT_DIR))
        logger.info(f'Saved (val_loss={avg_vloss:.4f})')

# Тест
logger.info("Testing:")
model.eval()
for i in range(min(5, len(val_subset))):
    item = val_subset[i]
    img = Image.open(item['image_path']).convert('RGB')
    pix = processor(img, return_tensors='pt').pixel_values.to(device)
    with torch.no_grad():
        gen = model.generate(pix, max_length=64, num_beams=4, early_stopping=True)
    pred = processor.batch_decode(gen, skip_special_tokens=True)[0]
    logger.info(f'  [{i+1}] GT=«{item["text"]}» → Pred=«{pred}»')

logger.info("Done!")