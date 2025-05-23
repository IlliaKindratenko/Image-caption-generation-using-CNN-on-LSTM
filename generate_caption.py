<<<<<<< HEAD
import torch
from torchvision import transforms
from PIL import Image
from model import EncoderCNN, DecoderRNN
import pickle
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(encoder_path="checkpoints/encoder_best.ckpt",
                decoder_path="checkpoints/decoder_best.ckpt",
                vocab_path="checkpoints/vocab.pkl"):

    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path) or not os.path.exists(vocab_path):
        raise FileNotFoundError("Не знайдені файли моделей або словника. Запусти спочатку тренування.")

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


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

def generate_caption(image_tensor, encoder, decoder, vocab):
    with torch.no_grad():
        features = encoder(image_tensor)
        sampled_ids = decoder.sample_beam_search(features, beam_size=3)

    sentence = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word in ("<sos>", "<pad>"):  # ігноруємо службові токени
            continue
        if word == "<eos>":
            break
        sentence.append(word)

    return " ".join(sentence)


def generate_caption_from_path(image_path, encoder, decoder, vocab):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    return generate_caption(image_tensor, encoder, decoder, vocab)


# Приклад використання
if __name__ == "__main__":
    image_path = "example/example4.jpg"  # Змініть шлях до свого зображення
    encoder, decoder, vocab = load_models()
    caption = generate_caption_from_path(image_path, encoder, decoder, vocab)
    print(f"📷 Опис зображення: {caption}")
=======
import torch
from torchvision import transforms
from PIL import Image
from model import EncoderCNN, DecoderRNN
import pickle
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(encoder_path="checkpoints/encoder_best.ckpt",
                decoder_path="checkpoints/decoder_best.ckpt",
                vocab_path="checkpoints/vocab.pkl"):

    if not os.path.exists(encoder_path) or not os.path.exists(decoder_path) or not os.path.exists(vocab_path):
        raise FileNotFoundError("Не знайдені файли моделей або словника. Запусти спочатку тренування.")

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


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

def generate_caption(image_tensor, encoder, decoder, vocab):
    with torch.no_grad():
        features = encoder(image_tensor)
        sampled_ids = decoder.sample_beam_search(features, beam_size=3)

    sentence = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word in ("<sos>", "<pad>"):  # ігноруємо службові токени
            continue
        if word == "<eos>":
            break
        sentence.append(word)

    return " ".join(sentence)


def generate_caption_from_path(image_path, encoder, decoder, vocab):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    return generate_caption(image_tensor, encoder, decoder, vocab)


# Приклад використання
if __name__ == "__main__":
    image_path = "example/example4.jpg"  # Змініть шлях до свого зображення
    encoder, decoder, vocab = load_models()
    caption = generate_caption_from_path(image_path, encoder, decoder, vocab)
    print(f"📷 Опис зображення: {caption}")
>>>>>>> d75ca5b77b9ca4ec7a8099c2e963b0f5cc4cd291
