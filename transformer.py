# %%
import torch
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torch import nn
import math

# %%
SRC = Field(tokenize="spacy", tokenizer_language="de", init_token="<sos>", eos_token="<eos>", lower=True)
TRG = Field(tokenize="spacy", tokenizer_language="en", init_token="<sos>", eos_token="<eos>", lower=True)
train_data, val_data, test_data = Multi30k.splits(exts=(".de", ".en"), fields=(SRC, TRG))


#%%
SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

# %%
device = "cpu"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128

train_iter, val_iter, test_iter = BucketIterator.splits(
    (train_data, val_data, test_data),
    batch_size=BATCH_SIZE,
    device=device
)

# %%
toy_vocab = torch.Tensor([[1, 2, 3]]).long() #[a,b,c]
#%%
D_MODEL = 512
P_DROP = 0.1
# %%
class Embeddings(nn.Module):
    def __init__(self, len_vocab, d_model=D_MODEL):
        
        super(Embeddings, self).__init__()
        self.embedding = nn.Embedding(len_vocab, D_MODEL)
        

    def forward(self, x):
        return self.embedding(x) * math.sqrt(D_MODEL)
        

# %%
toy_embedding_layer = Embeddings(toy_vocab.shape[-1]+1, d_model=4)
toy_embeddings = toy_embedding_layer(toy_vocab)
print(toy_embeddings)

# %%
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=D_MODEL, p_drop=P_DROP, max_len=5000):
        super().__init__()
        self.pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()

        two_i = torch.arange(0, d_model, step=2)
        div_term = torch.pow(10000, (two_i/d_model)).float()
        print(div_term.shape)        
        self.pe[:, 0::2] = torch.sin(pos/div_term)
        self.pe[:, 1::2] = torch.cos(pos/div_term)

        self.pe = self.pe.unsqueeze(0)
        self.register_buffer("pe", self.pe)

        self.dropout = nn.Dropout(P_DROP)
    # x is the input embedding
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)

# %%
toy_PE_layer = PositionalEncoding(d_model=4)
toy_PEs = toy_PE_layer(toy_embeddings)
print(toy_PEs)

# %%
class MultiHeadAttention():
    def __init__(self):
        pass
# %%
class AddNorm():
    def __init__(self):
        pass
# %%
class MLP():
    def __init__(self):
        pass
# %%
class EnocderLayer():
    def __init__(self):
        pass
# %%
class DecoderLayer():
    def __init__(self):
        pass
# %%
class Transformer(nn.Module):
    def __init__(self):
        pass        