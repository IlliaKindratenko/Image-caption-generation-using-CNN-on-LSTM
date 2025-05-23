<<<<<<< HEAD
import torch
from torchvision import transforms
from PIL import Image
from model import EncoderCNN, DecoderRNN
import pickle
import os
import nltk
from pycocotools.coco import COCO
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download('punkt')

# Налаштування
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Шляхи до даних і моделей
image_dir = "captions/val2014"
caption_path = "captions/annotations/captions_val2014.json"
vocab_path = "checkpoints/vocab.pkl"
encoder_path = "checkpoints/encoder_best.ckpt"
decoder_path = "checkpoints/decoder_best.ckpt"

# Завантаження словника
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

# Параметри моделі
embed_size = 256
hidden_size = 512
vocab_size = len(vocab)

# Ініціалізація моделей
encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size,
                     sos_token_id=vocab.stoi["<sos>"],
                     eos_token_id=vocab.stoi["<eos>"]).to(device)

encoder.load_state_dict(torch.load(encoder_path, map_location=device))
decoder.load_state_dict(torch.load(decoder_path, map_location=device))

encoder.eval()
decoder.eval()

# Трансформації зображень
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Завантаження COCO анотацій
coco = COCO(caption_path)
image_ids = coco.getImgIds()

smoothie = SmoothingFunction().method4

def generate_caption(image_tensor):
    with torch.no_grad():
        features = encoder(image_tensor)
        sampled_ids = decoder.sample_beam_search(features, beam_size=3)

    sentence = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == "<sos>":
            continue
        if word == "<eos>":
            break
        sentence.append(word)
    return sentence

# Оцінка на вибірці
total_bleu1 = 0
total_bleu4 = 0
sample_count = 100

print(f"🔎 Evaluating BLEU scores on {sample_count} images...")

for i, img_id in enumerate(image_ids[:sample_count]):
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    references = [nltk.word_tokenize(ann["caption"].lower()) for ann in anns]

    path = coco.loadImgs(img_id)[0]["file_name"]
    image = Image.open(os.path.join(image_dir, path)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    candidate = generate_caption(image_tensor)

    bleu1 = sentence_bleu(references, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu4 = sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    total_bleu1 += bleu1
    total_bleu4 += bleu4

    if i < 5:
        print(f"📷 Image {i+1}:\n  ➤ Generated: {' '.join(candidate)}\n  BLEU-1: {bleu1:.4f}, BLEU-4: {bleu4:.4f}\n")

avg_bleu1 = total_bleu1 / sample_count
avg_bleu4 = total_bleu4 / sample_count

print(f"✅ Середній BLEU-1: {avg_bleu1:.4f}")
print(f"✅ Середній BLEU-4: {avg_bleu4:.4f}")
=======
import torch
from torchvision import transforms
from PIL import Image
from model import EncoderCNN, DecoderRNN
import pickle
import os
import nltk
from pycocotools.coco import COCO
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download('punkt')

# Налаштування
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Шляхи до даних і моделей
image_dir = "captions/val2014"
caption_path = "captions/annotations/captions_val2014.json"
vocab_path = "checkpoints/vocab.pkl"
encoder_path = "checkpoints/encoder_best.ckpt"
decoder_path = "checkpoints/decoder_best.ckpt"

# Завантаження словника
with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

# Параметри моделі
embed_size = 256
hidden_size = 512
vocab_size = len(vocab)

# Ініціалізація моделей
encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size,
                     sos_token_id=vocab.stoi["<sos>"],
                     eos_token_id=vocab.stoi["<eos>"]).to(device)

encoder.load_state_dict(torch.load(encoder_path, map_location=device))
decoder.load_state_dict(torch.load(decoder_path, map_location=device))

encoder.eval()
decoder.eval()

# Трансформації зображень
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Завантаження COCO анотацій
coco = COCO(caption_path)
image_ids = coco.getImgIds()

smoothie = SmoothingFunction().method4

def generate_caption(image_tensor):
    with torch.no_grad():
        features = encoder(image_tensor)
        sampled_ids = decoder.sample_beam_search(features, beam_size=3)

    sentence = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == "<sos>":
            continue
        if word == "<eos>":
            break
        sentence.append(word)
    return sentence

# Оцінка на вибірці
total_bleu1 = 0
total_bleu4 = 0
sample_count = 100

print(f"🔎 Evaluating BLEU scores on {sample_count} images...")

for i, img_id in enumerate(image_ids[:sample_count]):
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)
    references = [nltk.word_tokenize(ann["caption"].lower()) for ann in anns]

    path = coco.loadImgs(img_id)[0]["file_name"]
    image = Image.open(os.path.join(image_dir, path)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    candidate = generate_caption(image_tensor)

    bleu1 = sentence_bleu(references, candidate, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu4 = sentence_bleu(references, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    total_bleu1 += bleu1
    total_bleu4 += bleu4

    if i < 5:
        print(f"📷 Image {i+1}:\n  ➤ Generated: {' '.join(candidate)}\n  BLEU-1: {bleu1:.4f}, BLEU-4: {bleu4:.4f}\n")

avg_bleu1 = total_bleu1 / sample_count
avg_bleu4 = total_bleu4 / sample_count

print(f"✅ Середній BLEU-1: {avg_bleu1:.4f}")
print(f"✅ Середній BLEU-4: {avg_bleu4:.4f}")
>>>>>>> d75ca5b77b9ca4ec7a8099c2e963b0f5cc4cd291
