import re
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import Counter, OrderedDict
from itertools import takewhile
from time import perf_counter
import sys

from bleu_eval import *

class VDS(torch.utils.data.Dataset):
    def __init__(self, training_path, label_path):
        super(VDS, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = pd.read_json(label_path)
        self._build_vocab()
        self._build_word_index()
        self._make_dataset(training_path)
        
    def _build_vocab(self):
        self.vocab = Counter()
        self.clength = 0
        for row in self.labels.itertuples(index=False):
            for fcaps in np.unique(np.array(row[0])):
                fcap_adj = re.sub(r'[^\w\s]', '', fcaps).lower().split(" ")
                self.vocab += Counter(fcap_adj)
                self.clength = len(fcap_adj) if len(fcap_adj) > self.clength else self.clength
        self.vocab = Counter(dict(filter(lambda i: i[1] > 3, self.vocab.items())))
        self.clength += 2
    
    def _build_word_index(self):
        # PAD is padding, BOS begin sentence, EOS end sentence, UNK unknown word
        self.tokens_idx = {0: "<PAD>", 1:"<BOS>", 2:"<EOS>", 3: "<UNK>"}
        self.tokens_word = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        for idx, w in enumerate(self.vocab):
            self.tokens_idx[idx+4] = w
            self.tokens_word[w] = idx+4
    
    def _make_dataset(self, training_path):
        self.input_feat = {}
        self.output_fcap_info = []
        for row in self.labels.itertuples(index=False):
            self.input_feat[row[1]] = torch.from_numpy(np.load(f"{training_path}/feat/{row[1]}.npy"))
            for fcaps in np.unique(np.array(row[0])):
                full_fcap = ["<BOS>"]
                for w in re.sub(r'[^\w\s]', '', fcaps).lower().split(" "):
                    if w in self.vocab:
                        full_fcap.append(w)
                    else:
                        full_fcap.append("<UNK>")
                full_fcap.append("<EOS>")
                full_fcap.extend(["<PAD>"]*(self.clength-len(full_fcap)))
                tokenized_fcap = [self.tokens_word[word] for word in full_fcap]
                
                # Index 0: Caption, Index 1: Full Cap w/ string info, Index 2: OneHotEncoded, Index 3: Video Label
                self.output_fcap_info.append([fcaps, full_fcap, tokenized_fcap, row[1]])
                
    def __len__(self):
        return len(self.output_fcap_info)
    
    def __getitem__(self, idx):
        caption, _, tokenized_fcap, video_label = self.output_fcap_info[idx]
        tensor_fcap = torch.Tensor(tokenized_fcap)
        one_hot = torch.nn.functional.one_hot(tensor_fcap.to(torch.int64), num_classes=len(self.tokens_idx))
        return self.input_feat[video_label], one_hot, caption

class TestVDS(torch.utils.data.Dataset):
    def __init__(self, test_path, test_label_path, vocab, tokens_word, tokens_idx, clength):
        super(TestVDS, self).__init__()
        self.vocab = vocab
        self.tokens_word = tokens_word
        self.tokens_idx = tokens_idx
        self.clength = clength
        self.labels = pd.read_json(test_label_path)
        self._make_dataset(test_path)
        
    def _make_dataset(self, training_path):
        self.input_feat = {}
        self.output_fcap_info = []
        for row in self.labels.itertuples(index=False):
            self.input_feat[row[1]] = torch.from_numpy(np.load(f"{training_path}/feat/{row[1]}.npy"))
            for fcaps in np.unique(np.array(row[0])):
                full_fcap = ["<BOS>"]
                for w in re.sub(r'[^\w\s]', '', fcaps).lower().split(" "):
                    if w in self.vocab:
                        full_fcap.append(w)
                    else:
                        full_fcap.append("<UNK>")
                full_fcap.append("<EOS>")
                full_fcap.extend(["<PAD>"]*(self.clength-len(full_fcap)))
                tokenized_fcap = [self.tokens_word[word] for word in full_fcap]
                
                # Index 0: Caption, Index 1: Full Cap w/ string info, Index 2: OneHotEncoded, Index 3: Video Label
                self.output_fcap_info.append([fcaps, full_fcap, tokenized_fcap, row[1]])
                
    def __len__(self):
        return len(self.output_fcap_info)
    
    def __getitem__(self, idx):
        caption, _, tokenized_fcap, video_label = self.output_fcap_info[idx]
        feat = self.input_feat[video_label]
        tensor_fcap = torch.Tensor(tokenized_fcap)
        one_hot = torch.nn.functional.one_hot(tensor_fcap.to(torch.int64), num_classes=len(self.tokens_idx))
        return feat, one_hot, caption, video_label

class S2VT(torch.nn.Module):
    def __init__(self, vocab_size, batch_size, frame_dim, hidden_dim, dropout, f_len, device, caption_length):
        super(S2VT, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.frame_dim = frame_dim
        self.hidden_dim = hidden_dim
        self.f_len = f_len
        self.caption_length = caption_length

        self.drop = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(frame_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)

        self.lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(2*hidden_dim, hidden_dim, batch_first=True)

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
    def forward(self, feat, cap):
        feat = self.linear1(self.drop(feat.contiguous().view(-1, self.frame_dim)))
        feat = feat.view(-1, self.f_len, self.hidden_dim)
        padding = torch.zeros([feat.shape[0], self.caption_length-1, self.hidden_dim]).to(self.device)
        feat = torch.cat((feat, padding), 1)
        
        l1, h1 = self.lstm1(feat)
        
        cap = self.embedding(cap[:,:self.caption_length-1])
        padding = torch.zeros([feat.shape[0], 80, self.hidden_dim]).to(self.device)
        cap = torch.cat((padding, cap), 1)
        cap = torch.cat((cap, l1), 2) # bs, 120, 1024

        l2, h2 = self.lstm2(cap) # 32, 120, 512
        
        l2 = l2[:, self.f_len:, :] # batch_size, 40, hidden
        l2 = self.drop(l2.contiguous().view(-1, self.hidden_dim)) # 1280, 512 (contig)
        output = F.log_softmax(self.linear2(l2), dim=1) # 1280, 2046
        return output
    
    def test(self, feat):
        caption = []
        feat = self.linear1(self.drop(feat.contiguous().view(-1, self.frame_dim)))
        feat = feat.view(-1, self.f_len, self.hidden_dim)
        padding = torch.zeros([feat.shape[0], self.caption_length-1, self.hidden_dim]).to(self.device)
        feat = torch.cat((feat, padding), 1)
        l1, h1 = self.lstm1(feat)
        
        padding = torch.zeros([feat.shape[0], self.caption_length-1, self.hidden_dim]).to(self.device)
        cap_in = torch.cat((padding, l1), 1)
        l2, h2 = self.lstm2(cap_in)
        
        bos = torch.ones(self.batch_size).to(self.device)
        cap_in = self.embedding(bos)
        cap_in = torch.cat((cap_in, l1[:,80,:]),1).view(self.batch_size, 1, 2*self.hidden_dim)
        
        l2, h2 = self.lstm2(cap_in, h2)
        l2 = torch.argmax(self.linear2(self.drop(l2.contiguous().view(-1, self.hidden_dim))),1)
        
        caption.append(l2)
        for i in range(self.f_len-2):
            cap_in = self.embedding(l2)
            cap_in = torch.cat((cap_in, l1[:, self.f_len+1+i, :]), 1)
            cap_in = cap_in.view(self.batch_size, 1, 2* self.hidden_dim)
            l2, h2 = self.lstm2(cap_in, h2)
            l2 = l2.contiguous().view(-1, self.hidden_dim)
            l2 = torch.argmax(self.linear2(self.drop(l2)),1)
            caption.append(l2)
        return caption

def trainer(mod, opt, dataset, batch_size, device, its, caption_length, vocab_size, epochs=1):
    mod.to(device)
    criterion = nn.NLLLoss()
    
    print(f"Starting training for {epochs} epochs")
    for i in range(epochs):
        st = perf_counter()
        mod.train()
        store_labels = []
        for idx, data in enumerate(dataset):
            if idx == its:
                break
            mod.zero_grad()
            feat = data[0].requires_grad_().to(device)
            labels = data[1].max(2)[1].to(device)
            predicted_labels = mod(feat.float(), labels)
            
            predicted_labels = predicted_labels.reshape(-1, caption_length-1, vocab_size)
            loss = 0
            for b in range(data[0].shape[0]):          
                loss += criterion(predicted_labels[b,:], labels[b,1:])
            loss.backward()
            opt.step()
            sys.stdout.write("\r")
            sys.stdout.write("[%-20s] %s %d%%" % ("="*int(20*(idx+1)/its), "Epoch Completion", 100*(idx+1)/its))
            sys.stdout.flush()
            
        print(f" Epochs: {i}, Loss: {loss}, Time Taken: {perf_counter()-st:.4f}")

def evaluator(mod, test_dataset, device, caption_length, vocab_size, detokenize_dict, output_filename):
    print("Testing Model")
    mod.eval()
    criterion = nn.NLLLoss()
    store_labels = []
    store_predicted_labels = []
    video_labels = []
    with torch.no_grad():
        for idx, data in enumerate(test_dataset):
            mod.zero_grad()
            feat = data[0].requires_grad_().to(device)
            labels = data[1].max(2)[1].to(device)
            predicted_labels = mod(feat.float(), labels)

            predicted_labels = predicted_labels.reshape(-1, caption_length-1, vocab_size)
            store_labels, store_predicted_labels = detoken(predicted_labels, data[2], store_labels, store_predicted_labels, detokenize_dict)
            
            loss = 0
            for b in range(data[0].shape[0]):          
                loss += criterion(predicted_labels[b,:], labels[b,1:])
                video_labels.append(data[3][b])
    w2f(video_labels, store_predicted_labels, store_labels, output_filename=output_filename)

def detoken(predicted_labels, labels, store_labels, store_predicted_labels, detokenize_dict):
    endsyntax = ["<EOS>", "<PAD>"]
    predicted_labels_index = predicted_labels.max(2)[1]
    for i in range(predicted_labels.shape[0]):
        predicted_label = [detokenize_dict[int(w_idx.numpy())] for w_idx in predicted_labels_index[i,:]]
        predicted_label = list(takewhile(lambda x: x not in endsyntax, predicted_label))
        
        store_labels.append(str(labels[i]))
        store_predicted_labels.append(" ".join(predicted_label))
    return store_labels, store_predicted_labels

def w2f(test_fname, predicted_labels, store_labels, output_filename="result.txt"):
    with open(output_filename, "w") as f:
        for i in range(len(store_labels)):
            f.write(f"{test_fname[i]}, {predicted_labels[i]}\n")

def bleu_score(output_filename="result.txt", correct_label_path="./MLDS_hw2_1_data/testing_label.json"):
    test = json.load(open(correct_label_path,'r'))
    result = {}
    with open("./"+output_filename,'r') as f:
        for line in f:
            line = line.rstrip()
            comma = line.index(',')
            test_id = line[:comma]
            caption = line[comma+1:]
            result[test_id] = caption
    bleu=[]
    for item in test:
        score_per_video = []
        captions = [x.rstrip('.') for x in item['caption']]
        score_per_video.append(BLEU(result[item['id']],captions,True))
        bleu.append(score_per_video[0])
    average = sum(bleu) / len(bleu)
    print("Average bleu score is " + str(average))