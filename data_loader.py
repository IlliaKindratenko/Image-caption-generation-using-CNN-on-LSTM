<<<<<<< HEAD
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import nltk
from pycocotools.coco import COCO
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    # Фільтруємо всі None
    batch = list(filter(lambda x: x is not None, batch))

    images = [item[0] for item in batch]
    captions = [item[1] for item in batch]

    images = torch.stack(images, 0)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)

    return images, captions

class CocoDataset(Dataset):
    def __init__(self, root, annotation_file, vocab, transform=None):
        self.root = root
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        ann_id = self.ids[index]
        caption = self.coco.anns[ann_id]['caption']
        image_id = self.coco.anns[ann_id]['image_id']
        image_file = self.coco.loadImgs(image_id)[0]['file_name']

        try:
            image = Image.open(os.path.join(self.root, image_file)).convert('RGB')
        except (FileNotFoundError, OSError) as e:
            print(f"Помилка при завантаженні {image_file}: {e}")
            return None
        if self.transform is not None:
            image = self.transform(image)

        tokens = nltk.tokenize.word_tokenize(str(caption).lower(), language="english")
        caption_idxs = [self.vocab.word2idx['<sos>']]
        caption_idxs += [self.vocab.word2idx.get(token, self.vocab.word2idx['<unk>']) for token in tokens]
        caption_idxs.append(self.vocab.word2idx['<eos>'])

        caption_tensor = torch.tensor(caption_idxs, dtype=torch.long)

        return image, caption_tensor

    def __len__(self):
        return len(self.ids)
>>>>>>> d75ca5b77b9ca4ec7a8099c2e963b0f5cc4cd291
