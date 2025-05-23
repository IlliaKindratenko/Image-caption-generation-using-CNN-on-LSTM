<<<<<<< HEAD
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torch
from torchvision import transforms
from model import EncoderCNN, DecoderRNN
import pickle

# Пристрій
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Завантаження моделі
def load_models(encoder_path="checkpoints/encoder_best.ckpt",
                decoder_path="checkpoints/decoder_best.ckpt",
                vocab_path="checkpoints/vocab.pkl"):
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)

    embed_size = 256
    hidden_size = 512
    vocab_size = len(vocab)

    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(
        embed_size,
        hidden_size,
        vocab_size,
        eos_token_id=vocab.stoi["<eos>"],
        sos_token_id=vocab.stoi["<sos>"]
    ).to(device)

    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    encoder.eval()
    decoder.eval()

    return encoder, decoder, vocab

# Преобразування
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

def generate_caption(image_path, encoder, decoder, vocab):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(image_tensor)
        sampled_ids = decoder.sample_beam_search(features, beam_size=3)

    words = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word in ("<sos>", "<pad>"):
            continue
        if word == "<eos>":
            break
        words.append(word)

    return " ".join(words)

# Головне вікно
class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Генерація опису зображення")
        self.root.geometry("700x700")
        self.root.configure(bg="#f5f5f5")

        self.encoder, self.decoder, self.vocab = load_models()
        self.image_path = None

        self.title_label = tk.Label(root, text="🖼️ Генерація опису до зображення",
                                    font=("Segoe UI", 18, "bold"), bg="#f5f5f5", fg="#333")
        self.title_label.pack(pady=20)

        self.select_button = tk.Button(root, text="📂 Вибрати зображення", font=("Segoe UI", 12),
                                       command=self.select_image, bg="#4CAF50", fg="white", width=25, height=2)
        self.select_button.pack(pady=10)

        self.image_label = tk.Label(root, bg="#ddd", width=300, height=300)
        self.image_label.pack(pady=10)

        self.caption_label = tk.Label(root, text="Опис зображення з'явиться тут", wraplength=600,
                                      font=("Segoe UI", 12), bg="#f5f5f5", fg="#555", justify="center")
        self.caption_label.pack(pady=10)

        self.evaluate_button = tk.Button(root, text="🧠 Згенерувати опис", font=("Segoe UI", 12),
                                         command=self.evaluate_image, bg="#2196F3", fg="white", width=25, height=2)
        self.evaluate_button.pack(pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if file_path:
            self.image_path = file_path
            image = Image.open(file_path)
            image = image.resize((300, 300))
            photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=photo)
            self.image_label.image = photo
            self.caption_label.config(text="⏳ Очікується генерація...")

    def evaluate_image(self):
        if self.image_path:
            caption = generate_caption(self.image_path, self.encoder, self.decoder, self.vocab)
            self.caption_label.config(text=f"📋 Опис: {caption}")
        else:
            self.caption_label.config(text="❗ Будь ласка, виберіть зображення.")

# Запуск програми
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()
>>>>>>> d75ca5b77b9ca4ec7a8099c2e963b0f5cc4cd291
