#!/usr/bin/env python3
"""Fine-tune TrOCR on historical Russian text data.

Использует 205,719 пар изображение-текст из data/images/ и data/texts/
для дообучения taiga75/ru-trocr-1700s.

На M2 Pro обучение ~5-8 часов на 30k samples.
"""

import json
import os
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, Features, Image as HFImage, Value
from loguru import logger
from PIL import Image as PILImage
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    VisionEncoderDecoderModel,
    default_data_collator,
)
from evaluate import load as load_metric

# ── Configuration ──
DATA_IMAGE_DIR = Path("data/images")
DATA_TEXT_DIR = Path("data/texts")
MODEL_NAME = "taiga75/ru-trocr-1700s"
OUTPUT_DIR = Path("app/models/ocr/fine_tuned_trocr")
MAX_SAMPLES = 50000  # Используем 50k из 205k для времени
VAL_SPLIT = 0.1
TEST_SPLIT = 0.05
MAX_LENGTH = 128  # Максимальная длина текста (токенов)
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
SAVE_STEPS = 500
EVAL_STEPS = 500

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ── 1. Load and prepare dataset ──
def scan_dataset(max_samples: int = MAX_SAMPLES):
    """Scan image-text pairs."""
    pairs = []
    image_files = sorted(os.listdir(DATA_IMAGE_DIR))

    for fname in image_files:
        if not (fname.endswith(".jpg") or fname.endswith(".png")):
            continue
        stem = Path(fname).stem
        txt_path = DATA_TEXT_DIR / f"{stem}.txt"
        if not txt_path.exists():
            continue
        img_path = DATA_IMAGE_DIR / fname
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        pairs.append((str(img_path), text))
        if len(pairs) >= max_samples:
            break

    logger.info(f"Found {len(pairs)} valid image-text pairs")
    return pairs


def create_dataset(pairs):
    """Create HuggingFace Dataset from pairs."""
    # Shuffle
    random.shuffle(pairs)

    n = len(pairs)
    n_val = int(n * VAL_SPLIT)
    n_test = int(n * TEST_SPLIT)
    n_train = n - n_val - n_test

    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train : n_train + n_val]
    test_pairs = pairs[n_train + n_val :]

    logger.info(
        f"Split: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}"
    )

    datasets = {}
    for split_name, split_pairs in [
        ("train", train_pairs),
        ("validation", val_pairs),
        ("test", test_pairs),
    ]:
        images = []
        texts = []
        for img_path, text in split_pairs:
            images.append(img_path)
            texts.append(text)

        ds = Dataset.from_dict(
            {"image": images, "text": texts},
            features=Features(
                {"image": HFImage(), "text": Value("string")}
            ),
        )
        datasets[split_name] = ds

    return DatasetDict(datasets)


# ── 2. Preprocessing ──
class TrOCRDatasetProcessor:
    """Process images and texts for TrOCR."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # TrOCR tokenizer uses eos_token as pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def preprocess(self, batch):
        """Process a batch of images + texts."""
        # Process images
        pixel_values = []
        for img_path in batch["image"]:
            try:
                image = PILImage.open(img_path).convert("RGB")
                processed = self.feature_extractor(
                    image, return_tensors="pt"
                ).pixel_values[0]
                pixel_values.append(processed)
            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
                # Use a blank image as fallback
                blank = PILImage.new("RGB", (384, 384), color="white")
                processed = self.feature_extractor(
                    blank, return_tensors="pt"
                ).pixel_values[0]
                pixel_values.append(processed)

        # Process texts
        tokens = self.tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        return {
            "pixel_values": torch.stack(pixel_values),
            "labels": tokens.input_ids,
            "attention_mask": tokens.attention_mask,
        }


def compute_metrics(pred, tokenizer):
    """Compute CER and WER metrics."""
    cer_metric = load_metric("cer")
    wer_metric = load_metric("wer")

    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer, "wer": wer}


# ── 3. Main training function ──
def train():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info("Scanning dataset...")
    pairs = scan_dataset()
    dataset = create_dataset(pairs)

    # Initialize processor
    processor = TrOCRDatasetProcessor()
    processor.feature_extractor.save_pretrained(str(OUTPUT_DIR))
    processor.tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Process datasets
    logger.info("Processing datasets...")
    processed_dataset = dataset.map(
        processor.preprocess,
        batched=True,
        batch_size=BATCH_SIZE,
        remove_columns=["image", "text"],
    )

    # Load model
    logger.info(f"Loading model: {MODEL_NAME}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)

    # Set model config for fine-tuning
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.config.max_length = MAX_LENGTH
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_steps=SAVE_STEPS,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to="none",
        fp16=False,  # MPS doesn't support fp16
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        tokenizer=processor.feature_extractor,
        data_collator=default_data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, processor.tokenizer),
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info("Saving model...")
    trainer.save_model(str(OUTPUT_DIR))
    processor.feature_extractor.save_pretrained(str(OUTPUT_DIR))
    processor.tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate(processed_dataset["test"])
    logger.info(f"Test results: {test_results}")

    # Save results
    with open(OUTPUT_DIR / "training_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    logger.info(f"Training complete! Model saved to {OUTPUT_DIR}")
    return test_results


if __name__ == "__main__":
    train()
