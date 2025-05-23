<<<<<<< HEAD
import pickle

vocab_path = "checkpoints/vocab.pkl"

# Завантаження
with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)

print("✅ Словник успішно завантажено.")
print(f"🔢 Розмір словника: {len(vocab)} токенів")

# Перевірка важливих токенів
for token in ["<pad>", "<sos>", "<eos>", "<unk>", "a", "man", "dog"]:
    idx = vocab.stoi.get(token, None)
    print(f"'{token}': {idx}")
>>>>>>> d75ca5b77b9ca4ec7a8099c2e963b0f5cc4cd291
