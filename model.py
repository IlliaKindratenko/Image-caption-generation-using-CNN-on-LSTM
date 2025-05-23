<<<<<<< HEAD
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        modules = list(effnet.children())[:-1]
        self.efficient = nn.Sequential(*modules)
        self.linear = nn.Linear(effnet.classifier[1].in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.efficient(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, sos_token_id=None, eos_token_id=None):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        hiddens = self.dropout(hiddens)
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, features, max_len=20):
        sampled_ids = []
        states = None

        if self.sos_token_id is None:
            raise ValueError("sos_token_id must be provided for generation.")

        input_token = torch.tensor([self.sos_token_id], device=features.device).unsqueeze(0)
        inputs = self.embed(input_token)

        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted.item())

            if self.eos_token_id is not None and predicted.item() == self.eos_token_id:
                break

            inputs = self.embed(predicted).unsqueeze(1)

        return sampled_ids

    def sample_beam_search(self, features, beam_size=3, max_len=20):
        sequences = [[[], 0.0, features.unsqueeze(1), None]]

        for _ in range(max_len):
            all_candidates = []
            for seq, score, input_seq, state in sequences:
                hiddens, state = self.lstm(input_seq, state)
                logits = self.linear(hiddens.squeeze(1))
                probs = torch.nn.functional.log_softmax(logits, dim=1)
                topk = torch.topk(probs, beam_size)

                for i in range(beam_size):
                    word_idx = topk.indices[0][i].item()
                    word_score = topk.values[0][i].item()
                    new_seq = seq + [word_idx]
                    new_score = score + word_score
                    next_input = self.embed(torch.tensor([word_idx], device=probs.device)).unsqueeze(1)
                    all_candidates.append([new_seq, new_score, next_input, state])

            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

            if all(s[0][-1] == self.eos_token_id for s in sequences if len(s[0]) > 0):
                break

        return sequences[0][0]
=======
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        modules = list(effnet.children())[:-1]
        self.efficient = nn.Sequential(*modules)
        self.linear = nn.Linear(effnet.classifier[1].in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.efficient(images)
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, sos_token_id=None, eos_token_id=None):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(inputs)
        hiddens = self.dropout(hiddens)
        outputs = self.linear(hiddens)
        return outputs

    def sample(self, features, max_len=20):
        sampled_ids = []
        states = None

        if self.sos_token_id is None:
            raise ValueError("sos_token_id must be provided for generation.")

        input_token = torch.tensor([self.sos_token_id], device=features.device).unsqueeze(0)
        inputs = self.embed(input_token)

        for _ in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            sampled_ids.append(predicted.item())

            if self.eos_token_id is not None and predicted.item() == self.eos_token_id:
                break

            inputs = self.embed(predicted).unsqueeze(1)

        return sampled_ids

    def sample_beam_search(self, features, beam_size=3, max_len=20):
        sequences = [[[], 0.0, features.unsqueeze(1), None]]

        for _ in range(max_len):
            all_candidates = []
            for seq, score, input_seq, state in sequences:
                hiddens, state = self.lstm(input_seq, state)
                logits = self.linear(hiddens.squeeze(1))
                probs = torch.nn.functional.log_softmax(logits, dim=1)
                topk = torch.topk(probs, beam_size)

                for i in range(beam_size):
                    word_idx = topk.indices[0][i].item()
                    word_score = topk.values[0][i].item()
                    new_seq = seq + [word_idx]
                    new_score = score + word_score
                    next_input = self.embed(torch.tensor([word_idx], device=probs.device)).unsqueeze(1)
                    all_candidates.append([new_seq, new_score, next_input, state])

            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

            if all(s[0][-1] == self.eos_token_id for s in sequences if len(s[0]) > 0):
                break

        return sequences[0][0]
>>>>>>> d75ca5b77b9ca4ec7a8099c2e963b0f5cc4cd291
