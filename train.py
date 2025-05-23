<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import EncoderCNN, DecoderRNN
from vocab import Vocabulary
import os
import json
import pickle
from data_loader import CocoDataset, collate_fn
from tqdm import tqdm

if __name__ == '__main__':
    # Параметри
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    learning_rate = 3e-4
    num_epochs = 20
    batch_size = 64
    freq_threshold = 5

    # Шляхи
    image_dir = 'captions/train2014'
    caption_path = 'captions/annotations/captions_train2014.json'
    save_dir = 'checkpoints'
    os.makedirs(save_dir, exist_ok=True)

    # Пристрій
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Побудова словника
    with open(caption_path, 'r') as f:
        captions = json.load(f)
    all_captions = [ann['caption'] for ann in captions['annotations']]
    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(all_captions)

    # Збереження словника
    vocab_path = os.path.join(save_dir, "vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    print(f"✅ Словник збережено у файл {vocab_path}")

    # Трансформації
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Датасет і DataLoader
    dataset = CocoDataset(
        root=image_dir,
        annotation_file=caption_path,
        vocab=vocab,
        transform=transform
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Моделі
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(
        embed_size, hidden_size, len(vocab), num_layers,
        eos_token_id=vocab.stoi["<eos>"],
        sos_token_id=vocab.stoi["<sos>"]
    ).to(device)

    # Оптимізатор і функція втрат
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)

    # Scheduler: ReduceLROnPlateau, зменшує LR при зупинці покращення loss
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2
    )

    # Навчання
    best_loss = float('inf')
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0

        for idx, (imgs, captions) in enumerate(tqdm(data_loader, desc="Training", leave=True)):
            imgs = imgs.to(device)
            captions = captions.to(device)

            features = encoder(imgs)
            outputs = decoder(features, captions)

            loss = criterion(outputs.view(-1, outputs.size(2)), captions.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}\n")

        # Збереження найкращої моделі
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), os.path.join(save_dir, "encoder_best.ckpt"))
            torch.save(decoder.state_dict(), os.path.join(save_dir, "decoder_best.ckpt"))
            print(f"✅ Збережено найкращу модель з loss: {best_loss:.4f}")

        # Збереження моделей поточної епохи
        encoder_path = os.path.join(save_dir, f"encoder_epoch{epoch+1}.ckpt")
        decoder_path = os.path.join(save_dir, f"decoder_epoch{epoch+1}.ckpt")
        torch.save(encoder.state_dict(), encoder_path)
        torch.save(decoder.state_dict(), decoder_path)
        print(f"✅ Моделі збережено: {encoder_path}, {decoder_path}")

        # Оновлення scheduler на основі avg_loss
        scheduler.step(avg_loss)
>>>>>>> d75ca5b77b9ca4ec7a8099c2e963b0f5cc4cd291
