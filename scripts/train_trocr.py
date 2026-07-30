#!/usr/bin/env python3
"""
Fine-tuning TrOCR для исторических русских текстов
Использует MPS (Metal GPU) на M-chip Mac
"""
import json
import torch
from pathlib import Path
from PIL import Image
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel,
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments,
)

# Настройки
DATA_DIR = Path("finetune_data")
OUTPUT_DIR = Path("trocr-finetuned")
OUTPUT_DIR.mkdir(exist_ok=True)

# Проверяем доступность MPS
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f" Используем устройство: {device}")
print(f" PyTorch версия: {torch.__version__}")

# Загрузка данных из JSONL
def load_jsonl(path):
    """Загрузка данных из JSONL файла"""
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data

print("\nЗагрузка данных...")
train_data = load_jsonl(DATA_DIR / "train.jsonl")
val_data = load_jsonl(DATA_DIR / "val.jsonl")
test_data = load_jsonl(DATA_DIR / "test.jsonl")

print(f" Train: {len(train_data)} образцов")
print(f" Validation: {len(val_data)} образцов")
print(f" Test: {len(test_data)} образцов")

# Загрузка процессора и модели TrOCR
print("\n Загрузка модели TrOCR-base-printed...")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

# Конфигурация модели
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id

# Кастомный Data Collator для TrOCR
class DataCollatorForOCRDataset:
    """Кастомный Data Collator для обработки изображений и текста"""
    
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, features):
        # Извлекаем и стакаем pixel_values и labels
        pixel_values = torch.stack([feature["pixel_values"] for feature in features])
        labels = torch.stack([feature["labels"] for feature in features])
        
        # Заменяем pad_token_id на -100, чтобы loss игнорировал padding
        labels = torch.where(labels == self.processor.tokenizer.pad_token_id, -100, labels)
        
        return {
            "pixel_values": pixel_values,
            "labels": labels,
        }

# Датасет для PyTorch
class OCRDataset(torch.utils.data.Dataset):
    """Кастомный датасет для OCR"""
    
    def __init__(self, data, processor, max_length=128):
        self.data = data
        self.processor = processor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Загрузка и обработка изображения
        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.processor.image_processor(
            image, 
            return_tensors="pt"
        ).pixel_values.squeeze()
        
        # Токенизация текста
        encoding = self.processor.tokenizer(
            item["text"], 
            padding="max_length", 
            max_length=self.max_length, 
            truncation=True,
            return_tensors="pt"
        )
        
        labels = encoding.input_ids.squeeze()
        
        return {
            "pixel_values": pixel_values, 
            "labels": labels
        }

# Создаем датасеты
print("\n Подготовка датасетов...")
train_dataset = OCRDataset(train_data, processor)
val_dataset = OCRDataset(val_data, processor)
test_dataset = OCRDataset(test_data, processor)

# Создаем Data Collator
data_collator = DataCollatorForOCRDataset(processor)

# Аргументы обучения
training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=2,  # Уменьшите если не хватает памяти
    per_device_eval_batch_size=8,
    learning_rate=1e-5,
    num_train_epochs=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_steps=50,
    dataloader_num_workers=4,
    predict_with_generate=True,
    generation_max_length=128,
    fp16=False,  # MPS не поддерживает fp16 хорошо
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    dataloader_pin_memory=False,  # Исправляем предупреждение о pin_memory на MPS
    report_to="none",  # Отключаем wandb
)

# Создаем Trainer
print("\n Создание Trainer...")
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    processing_class=processor.tokenizer,  # Используем processing_class вместо tokenizer
)

# Обучение
print("\nНачинаем обучение...")
print(f" Эпохи: {training_args.num_train_epochs}")
print(f" Batch size: {training_args.per_device_train_batch_size}")
print(f" Learning rate: {training_args.learning_rate}")

try:
    trainer.train()
    print("\nОбучение завершено!")
except KeyboardInterrupt:
    print("\nОбучение прервано пользователем")
    print("Сохраняем текущую модель...")

# Сохранение модели
final_model_dir = OUTPUT_DIR / "final"
final_model_dir.mkdir(exist_ok=True)

print(f"\nСохранение модели в {final_model_dir}...")
model.save_pretrained(final_model_dir)
processor.save_pretrained(final_model_dir)

print("\nМодель успешно сохранена!")
print(f"\nПуть к модели: {final_model_dir.absolute()}")

# Быстрый тест на val наборе
print("\nТестирование на validation наборе...")
model.eval()
sample = val_data[0]
image = Image.open(sample["image_path"]).convert("RGB")
pixel_values = processor(image, return_tensors="pt").pixel_values

with torch.no_grad():
    generated_ids = model.generate(pixel_values)
    predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"\nПример предсказания:")
print(f"   Ground truth: {sample['text']}")
print(f"   Предсказание: {predicted_text}")

print("\nВсе готово! Модель можно использовать в пайплайне OCR.")
