import torch

from utils import *

training_data_path = "./MLDS_hw2_1_data/training_data/"
training_labels = "./MLDS_hw2_1_data/training_label.json"

ds = VDS(training_data_path, training_labels) # Length of caption: 40+2 for bos, eos , items in vocab=2406

batch_size = 64
hidden_dim = 256
vocab_size = len(ds.tokens_idx)
detokenize_dict = ds.tokens_idx
feat_size = 4096
seq_length = 80
caption_length = ds.clength
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
drop = 0.3
lr=1e-4

epochs = 100
its = len(ds) // batch_size

mod = S2VT(vocab_size, batch_size, feat_size, hidden_dim, drop, 80, device, caption_length)
dataset = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
opt = torch.optim.Adam(mod.parameters(), lr=lr)

print(f"Training for {batch_size} bs, {epochs} epochs, {lr} lr, {hidden_dim} hid dim")
trainer(
    mod=mod, 
    opt=opt, 
    dataset=dataset, 
    batch_size=batch_size, 
    device=device, 
    its=its, 
    caption_length=caption_length, 
    vocab_size=vocab_size, 
    epochs=epochs
)

path = f"./m_save_{epochs}e_{batch_size}bs_{lr}_{hidden_dim}.pth"
torch.save(mod.state_dict(), path)