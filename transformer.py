# %%
import torch
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
from torch import nn, Tensor
import torch.nn.functional as F
import math
from torchsummary import summary

# %%
SRC = Field(tokenize="spacy", tokenizer_language="de",
            init_token="<sos>", eos_token="<eos>", lower=True)
TRG = Field(tokenize="spacy", tokenizer_language="en",
            init_token="<sos>", eos_token="<eos>", lower=True)
train_data, val_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(SRC, TRG))


# %%
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
toy_vocab = torch.Tensor([[1, 2, 3]]).long()  # [a,b,c]
# %%
D_MODEL = 512
P_DROP = 0.1
# %%


class Embeddings(nn.Module):
    def __init__(self, len_vocab, d_model=D_MODEL):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(len_vocab, self.d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


# %%
toy_embedding_layer = Embeddings(toy_vocab.shape[-1]+1, d_model=4)
toy_embeddings = toy_embedding_layer(toy_vocab)
print(toy_embeddings)

# %%


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=D_MODEL, p_drop=P_DROP, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()

        two_i = torch.arange(0, d_model, step=2)
        div_term = torch.pow(10000, (two_i/d_model)).float()
        pe[:, 0::2] = torch.sin(pos/div_term)
        pe[:, 1::2] = torch.cos(pos/div_term)

        pe = pe.unsqueeze(0)

        # assigns the first argument to a class variable
        # i.e. self.pe
        self.register_buffer("pe", pe)

        self.dropout = nn.Dropout(P_DROP)

    # x is the input embedding
    def forward(self, x):

        # work through this line :S
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)


# %%
toy_PE_layer = PositionalEncoding(d_model=4)
toy_PEs = toy_PE_layer(toy_embeddings)
print(toy_PEs)

# %%


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=D_MODEL, num_heads=8, mask=None, p_drop=P_DROP):
        super().__init__()

        # d_q, d_k, d_v
        self.d = d_model//num_heads

        self.d_model = d_model
        self.num_heads = num_heads
        self.mask = mask

        self.dropout = nn.Dropout(P_DROP)

    def scaled_dot_product_attention(self, Q: Tensor, K: Tensor, V: Tensor):
        Q_K_matmul = torch.matmul(Q, K.transpose(-2, -1))
        matmul_scaled = Q_K_matmul/math.sqrt(self.d)

        if self.mask is None:
            scaled_softmax = F.softmax(matmul_scaled, dim=-1)

        dropout_scaled_softmax = self.dropout(scaled_softmax)
        return torch.matmul(dropout_scaled_softmax, V)

    def forward(self, x: Tensor, queries: Tensor = None, keys: Tensor = None, values: Tensor = None):
        q = x if not queries else queries
        k = x if not keys else keys
        v = x if not values else values

        Q = [nn.Linear(x.size(-1), self.d)(q)
             for head in range(self.num_heads)]
        K = [nn.Linear(x.size(-1), self.d)(k)
             for head in range(self.num_heads)]
        V = [nn.Linear(x.size(-1), self.d)(v)
             for head in range(self.num_heads)]

        scores_per_head = [self.scaled_dot_product_attention(
            Q_, K_, V_) for Q_, K_, V_ in zip(Q, K, V)]

        concat_scores = torch.cat(scores_per_head, -1)
        return nn.Linear(concat_scores.size(-1), self.d_model)(concat_scores)


# %%
toy_MHA_layer = MultiHeadAttention(d_model=4, num_heads=2)
toy_MHA = toy_MHA_layer(toy_PEs)
print(toy_MHA, toy_MHA.shape)
# %%


class AddNorm(nn.Module):
    def __init__(self, d_model=D_MODEL, p_drop=P_DROP):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x, res_input):
        ln = self.layer_norm(res_input + x)
        return self.dropout(ln)


# %%
toy_AddNorm_layer = AddNorm(d_model=4)
toy_AddNorm = toy_AddNorm_layer(toy_MHA, toy_PEs)
print(toy_AddNorm, toy_AddNorm.shape)

# %%


class PointwiseFeedforward(nn.Module):
    def __init__(self, d_model=D_MODEL, d_ff=2048, p_drop=P_DROP):
        super().__init__()
        self.pffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.pffn(x)


# %%
toy_PFFN_layer = PointwiseFeedforward(d_model=4, d_ff=16)
toy_PFFN = toy_PFFN_layer(toy_AddNorm)
print(toy_PFFN, toy_PFFN.shape)

# %%
toy_AddNorm_layer_2 = AddNorm(d_model=4)
toy_AddNorm_2 = toy_AddNorm_layer_2(toy_PFFN, toy_AddNorm)
print(toy_AddNorm_2, toy_AddNorm_2.shape)
# %%


class EncoderLayer(nn.Module):
    def __init__(self, d_model=D_MODEL, num_heads=8, d_ff=2048, p_drop=P_DROP):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.p_drop = p_drop

    def forward(self, x):
        mha = MultiHeadAttention(self.d_model, self.num_heads, p_drop=self.p_drop)(x)
        addNorm_1 = AddNorm(self.d_model, self.p_drop)(mha, x)
        pffn = PointwiseFeedforward(
            self.d_model, self.d_ff, self.p_drop)(addNorm_1)
        addNorm_2 = AddNorm(self.d_model, self.p_drop)(pffn, addNorm_1)
        return addNorm_2

# %%


class Encoder(nn.Module):
    def __init__(self, num_layers, len_vocab, d_model, num_heads, d_ff, p_drop):
        super().__init__()

        self.len_vocab = len_vocab
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.p_drop = p_drop

        self.encoders = nn.ModuleList([EncoderLayer(
            self.d_model,
            self.num_heads,
            self.d_ff,
            self.p_drop
        )] * num_layers)
        self.encodersModelStack = nn.Sequential(*self.encoders)

    def forward(self, x):
        embeddings = Embeddings(self.len_vocab, self.d_model)(x)
        positional_encoding = PositionalEncoding(
            self.d_model, self.p_drop)(embeddings)
        return self.encodersModelStack(positional_encoding)


# %%
toy_encoder = Encoder(3, 4, 4, 2, 16, 0.1)
toy_encoder_output = toy_encoder(toy_vocab)
print(toy_encoder_output, toy_encoder_output.shape)

# %%


class DecoderLayer():
    def __init__(self):
        pass
# %%


class Transformer(nn.Module):
    def __init__(self):
        pass
