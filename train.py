import os
import json
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


def main():
    # --------- إعدادات ---------
    DATA_DIR = "data"   # غيّرها إذا اسم مجلد بياناتك مختلف
    MODEL_OUT = os.path.join(DATA_DIR, "model.pt")
    CLASS_JSON = os.path.join(DATA_DIR, "class_names.json")
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LR = 1e-4
    IMG_SIZE = 224
    NUM_WORKERS = 0  # خليه 0 في Windows لتجنب مشاكل multiprocessing
    # ---------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Dataset مخصص
    class ImageLabelDataset(Dataset):
        def __init__(self, root_dir, split="train", transform=None):
            self.root = os.path.join(root_dir, split)
            self.images_dir = os.path.join(self.root, "images")
            self.labels_dir = os.path.join(self.root, "labels")
            self.transform = transform

            self.img_files = sorted([
                fn for fn in os.listdir(self.images_dir)
                if fn.lower().endswith((".jpg", ".jpeg", ".png"))
            ])

            self.label_texts = []
            for fn in self.img_files:
                base = os.path.splitext(fn)[0]
                label_path = os.path.join(self.labels_dir, base + ".txt")
                with open(label_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                # 🔹 نأخذ أول class_id فقط للتصنيف
                first_class = lines[0].strip().split()[0]
                self.label_texts.append(first_class)

        def __len__(self):
            return len(self.img_files)

        def __getitem__(self, idx):
            img_path = os.path.join(self.images_dir, self.img_files[idx])
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, self.label_texts[idx]

    # بناء الأصناف
    def build_label_map(dataset):
        names = sorted(list(set(dataset.label_texts)))
        return names, {n: i for i, n in enumerate(names)}

    def text_to_index_batch(batch_texts, name_to_idx):
        return torch.tensor([name_to_idx[t] for t in batch_texts], dtype=torch.long)

    # تحويلات الصور
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # تحميل البيانات
    train_ds = ImageLabelDataset(DATA_DIR, "train", transform=train_transform)
    val_ds = ImageLabelDataset(DATA_DIR, "val", transform=val_transform)

    class_names, name_to_idx = build_label_map(train_ds)
    print("Classes:", class_names)

    with open(CLASS_JSON, "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKERS)

    # النموذج
    model = models.resnet18(pretrained=True)
    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        # --- تدريب ---
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for imgs, texts in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            imgs = imgs.to(device)
            labels = text_to_index_batch(texts, name_to_idx).to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += imgs.size(0)

        print(f"Train Loss: {train_loss/train_total:.4f} Acc: {train_correct/train_total:.4f}")

        # --- تحقق ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, texts in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                imgs = imgs.to(device)
                labels = text_to_index_batch(texts, name_to_idx).to(device)

                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"✅ Saved best model to {MODEL_OUT}")

    print("Training done. Best val loss:", best_val_loss)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
