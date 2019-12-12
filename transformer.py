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
toy_embeddings = Embeddings(toy_vocab.shape[-1]+1, D_MODEL==4)
toy_embeddings(toy_vocab)

# %%
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=D_MODEL, p_drop=P_DROP):
        super().__init__()
        # sel
    
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