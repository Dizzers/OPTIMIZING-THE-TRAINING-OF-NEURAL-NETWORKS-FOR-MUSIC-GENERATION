import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
import pretty_midi
from torch import nn
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-3
DEVICE = "mps" if torch.mps.is_available() else "cpu"
OPTIMIZERS_TO_TEST = ["Adam", "AdamW", "RMSprop"]
RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"
SEQ_LEN = 128
MIN_NOTE = 24
MAX_NOTE = 108

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

pl.seed_everything(42)

class MIDIDataset(Dataset):
    def __init__(self, root_dir, seq_len=SEQ_LEN, min_note=MIN_NOTE, max_note=MAX_NOTE):
        self.files = glob.glob(os.path.join(root_dir, "**/*.mid"), recursive=True)
        self.files += glob.glob(os.path.join(root_dir, "**/*.midi"), recursive=True)

        if not self.files:
            raise RuntimeError(f"Не найдено MIDI файлов в: {root_dir}")

        self.seq_len = seq_len
        self.min_note = min_note
        self.max_note = max_note
        self.n_notes = max_note - min_note + 1

        print(f"[MIDIDataset] найдено MIDI файлов: {len(self.files)}")

    def midi_to_roll(self, path):
        try:
            pm = pretty_midi.PrettyMIDI(path)
        except Exception:
            print("Ошибка чтения MIDI:", path)
            return None

        fs = 4
        roll = pm.get_piano_roll(fs=fs).T
        roll = roll[:, self.min_note:self.max_note + 1]
        roll = (roll > 0).astype(np.float32)

        if len(roll) >= self.seq_len:
            roll = roll[:self.seq_len]
        else:
            pad = np.zeros((self.seq_len - len(roll), self.n_notes), dtype=np.float32)
            roll = np.vstack([roll, pad])

        return roll

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        roll = self.midi_to_roll(self.files[idx])
        if roll is None:
            return self.__getitem__((idx + 1) % len(self))

        x = torch.tensor(roll, dtype=torch.float32)
        y = torch.roll(x, shifts=-1, dims=0)
        return x, y

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        ce_loss = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        p_t = probs * targets + (1 - probs) * (1 - targets)

        focal_factor = (1 - p_t) ** self.gamma

        loss = self.alpha * focal_factor * ce_loss

        return loss.mean()

class BCELossWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.loss(logits, targets)

class SmallMusicModel(pl.LightningModule):
    def __init__(self, n_notes=MAX_NOTE-MIN_NOTE+1, hidden=128, lr=LR, optimizer_name="Adam"):
        super().__init__()
        self.save_hyperparameters()

        self.conv = nn.Sequential(
            nn.Conv1d(n_notes, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.gru = nn.GRU(64, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_notes)
        self.loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        out, _ = self.gru(x)
        return self.out(out)

    def compute_metrics(self, logits, targets):
        probs = torch.sigmoid(logits)
        threshold = 0.2
        preds = (probs > threshold).float()

        y_true = targets.cpu().numpy()
        y_pred = preds.cpu().numpy()

        y_true = y_true.reshape(-1, y_true.shape[-1])
        y_pred = y_pred.reshape(-1, y_pred.shape[-1]) 

        acc = (y_true == y_pred).mean()

        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

        bce_loss = self.loss_fn(logits, targets).item()
        perplexity = np.exp(bce_loss)

        return acc, f1, prec, rec, perplexity

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        acc, f1, prec, rec, perp = self.compute_metrics(logits, y)

        self.log_dict({
            "train_loss": loss,
            "train_acc": acc,
            "train_f1": f1,
            "train_precision": prec,
            "train_recall": rec,
            "train_perplexity": perp
        }, prog_bar=True, on_step=False, on_epoch=True) 

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        acc, f1, prec, rec, perp = self.compute_metrics(logits, y)

        self.log_dict({
            "val_loss": loss,
            "val_acc": acc,
            "val_f1": f1,
            "val_precision": prec,
            "val_recall": rec,
            "val_perplexity": perp
        }, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        name = self.hparams.optimizer_name
        if name == "Adam":
            return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        elif name == "AdamW":
            return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        elif name == "RMSprop":
            return torch.optim.RMSprop(self.parameters(), lr=self.hparams.lr)
        else:
            raise ValueError(f"Неизвестный оптимизатор: {name}")

def train_once(opt_name, train_loader, val_loader):
    logger = CSVLogger("logs", name=f"exp_{opt_name}")
    checkpoint = ModelCheckpoint(
        dirpath=f"{CHECKPOINT_DIR}/{opt_name}",
        save_top_k=1,
        monitor="val_loss",
        mode="min"
    )
    progress = TQDMProgressBar(refresh_rate=20)

    n_notes = train_loader.dataset.dataset.n_notes if isinstance(train_loader.dataset, torch.utils.data.Subset) else train_loader.dataset.n_notes
    model = SmallMusicModel(n_notes=n_notes, lr=LR, optimizer_name=opt_name)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1 if torch.mps.is_available() else None,
        max_epochs=EPOCHS,
        callbacks=[checkpoint, progress],
        logger=logger
    )

    trainer.fit(model, train_loader, val_loader)

    return {
        "optimizer": opt_name,
        "best_ckpt": checkpoint.best_model_path,
        "metrics_path": os.path.join(logger.log_dir, "metrics.csv")
    }

def save_prediction_as_midi(tensor, filename, min_note=MIN_NOTE):
    arr = torch.sigmoid(tensor).cpu().numpy()
    arr = arr > 0.5

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)

    time_step = 0.25
    T, N = arr.shape

    for note in range(N):
        active = False
        start = 0
        for t in range(T):
            if arr[t, note] and not active:
                active = True
                start = t * time_step
            if not arr[t, note] and active:
                active = False
                end = t * time_step
                inst.notes.append(pretty_midi.Note(
                    velocity=100,
                    pitch=min_note + note,
                    start=start,
                    end=end
                ))
        if active:
            inst.notes.append(pretty_midi.Note(
                velocity=100,
                pitch=min_note + note,
                start=start,
                end=T * time_step
            ))
    pm.instruments.append(inst)
    pm.write(filename)
    print("Saved:", filename)

def plot_and_save_metrics(df_all, metric_names):
    for metric in metric_names:
        plt.figure(figsize=(10,6))
        sns.lineplot(data=df_all, x="epoch", y=metric, hue="optimizer", marker="o")
        plt.title(f"{metric} per epoch for all optimizers")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{metric}.png"))
        plt.close()
    print("Все графики сохранены в папку:", RESULTS_DIR)

def compare_bce_vs_focal(train_loader, val_loader):
    print("\n============================")
    print("  Сравнение BCE и Focal Loss")
    print("============================")

    losses = {
        "BCE": BCELossWrapper(),
        "Focal": FocalLoss(alpha=0.25, gamma=2.0)
    }

    results = []

    for loss_name, criterion in losses.items():
        model = SmallMusicModel(n_notes=train_loader.dataset.dataset.n_notes,
                                optimizer_name="Adam")
        model.loss_fn = criterion

        logger = CSVLogger("logs", name=f"loss_{loss_name}")
        trainer = pl.Trainer(
            accelerator="auto",
            devices=1 if torch.mps.is_available() else None,
            max_epochs=EPOCHS,
            callbacks=[TQDMProgressBar(refresh_rate=20)],
            logger=logger
        )

        trainer.fit(model, train_loader, val_loader)

        metrics_path = os.path.join(logger.log_dir, "metrics.csv")
        results.append((loss_name, metrics_path))

    plt.figure(figsize=(10,6))

    for name, csv_path in results:
        df = pd.read_csv(csv_path)
        df["epoch"] = df.index + 1

        if "val_f1" in df.columns:
            sns.lineplot(data=df, x="epoch", y="val_f1", label=name, marker="o")

    plt.title("F1-score: BCE vs Focal Loss (Adam)")
    plt.xlabel("Epoch")
    plt.ylabel("F1-score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "BCE_vs_Focal_F1.png"))
    plt.close()

    print("График BCE_vs_Focal_F1.png сохранён!")
    
def run_experiments(midi_dir):
    dataset = MIDIDataset(midi_dir)
    val_size = int(0.1 * len(dataset))

    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [len(dataset)-val_size, val_size]
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    summary = []

    for opt in OPTIMIZERS_TO_TEST:
        print("\n============================")
        print("  Training:", opt)
        print("============================")
        res = train_once(opt, train_loader, val_loader)
        summary.append(res)

    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    print("Сводная таблица по оптимизаторам:")
    print(df_summary)

    df_all_list = []
    metric_list = ["train_loss","val_loss","train_acc","val_acc",
                   "train_f1","val_f1","train_precision","val_precision",
                   "train_recall","val_recall","train_perplexity","val_perplexity"]

    for opt in OPTIMIZERS_TO_TEST:
        paths = glob.glob(f"logs/exp_{opt}/**/metrics.csv", recursive=True)
        if paths:
            metrics_path = sorted(paths)[-1] 
            df_metrics = pd.read_csv(metrics_path)
            df_metrics["optimizer"] = opt
            df_metrics["epoch"] = df_metrics.index + 1

            existing_metrics = [m for m in metric_list if m in df_metrics.columns]

            if not existing_metrics:
                print(f"В metrics.csv для {opt} нет метрик")
                continue

            df_all_list.append(df_metrics[["epoch", "optimizer"] + existing_metrics])

    if not df_all_list:
        print("Ошибка: не найдено ни одного metrics.csv. Проверьте логи тренировки.")
        return df_summary

    df_all = pd.concat(df_all_list, ignore_index=True)
    df_all.to_csv(os.path.join(RESULTS_DIR, "all_metrics.csv"), index=False)
    print("Все метрики сохранены в CSV: all_metrics.csv")

    for metric in metric_list:
        plt.figure(figsize=(10,6))
        sns.lineplot(data=df_all, x="epoch", y=metric, hue="optimizer", marker="o")
        plt.title(f"{metric} per epoch for all optimizers")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{metric}.png"))
        plt.close()

    print("Все графики сохранены в папку:", RESULTS_DIR)
    print("Все графики сохранены в папку:", RESULTS_DIR)
    
    compare_bce_vs_focal(train_loader, val_loader)

    return df_summary

if __name__ == "__main__":
    DATASET_PATH = "archive" 
    df = run_experiments(DATASET_PATH)
    print("Эксперимент завершен, результаты сохранены в папку", RESULTS_DIR)
