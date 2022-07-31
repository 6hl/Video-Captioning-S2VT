import torch

from utils import *

torch.manual_seed(14)

data_location = sys.argv[1]
output_filename = sys.argv[2]

test_data_path = f"{str(data_location)}/testing_data/"
test_labels = str(f"{str(data_location)}/testing_label.json")
ds = torch.load("./vocab.dat")

batch_size = 10
hidden_dim = 256
vocab_size = len(ds.tokens_idx)
detokenize_dict = ds.tokens_idx
feat_size = 4096
seq_length = 80
caption_length = ds.clength
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")
drop = 0.5
lr=1e-4

its = len(ds) // batch_size

mod = S2VT(vocab_size, batch_size, feat_size, hidden_dim, drop, 80, device, caption_length)
test_ds = TestVDS(test_data_path, test_labels, ds.vocab, ds.tokens_word, ds.tokens_idx, ds.clength)
test_dataset = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)
opt = torch.optim.Adam(mod.parameters(), lr=lr)

mod.load_state_dict(torch.load("./model.pth"))
evaluator(mod, test_dataset, device, caption_length, vocab_size, detokenize_dict, output_filename)
bleu_score(output_filename=output_filename, correct_label_path=test_labels)