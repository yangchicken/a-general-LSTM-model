import math
import pandas as pd
import torch
from torch import nn
import os
import hashlib
import requests
import zipfile
import tarfile
import collections
from torch.utils import data
from matplotlib_inline import backend_inline
from IPython import display
from matplotlib import pyplot as plt
import time
import numpy as np

'''
    Transformer implementation from scratch using pytorch
'''

class DataLoad():
    def __init__(self, DATA_HUB = None):
        self.DATA_HUB = DATA_HUB

    def download(self, name, cache_dir = os.path.join('.', 'data')):
        """Download a file inserted into DATA_HUB, return the local filename."""
        assert name in self.DATA_HUB, f"{name} does not exist in {self.DATA_HUB}"
        url, sha1_hash = self.DATA_HUB[name]
        os.makedirs(cache_dir, exist_ok=True)
        fname = os.path.join(cache_dir, url.split('/')[-1])
        # 1. Check if the file is already exists 
        if os.path.exists(fname):
            # 2. Calculate the SHA1 hash of the file
            sha1 = hashlib.sha1()
            with open(fname, 'rb') as f:
                while True:
                    data = f.read(1048576)  # Read 1MB each time 
                    if not data:
                        break
                    sha1.update(data)
            # 3. Verify that hashes match
            if sha1.hexdigest() == sha1_hash:
                return fname  # Cache hits, return directly
        # 4. Cache miss or file does not exist, download file
        print(f'Downloading {fname} from {url}...')
        r = requests.get(url, stream=True, verify=True)
        with open(fname, 'wb') as f:
            f.write(r.content)
        return fname
    
    def download_extract(self, name, folder = None):
        """Download and extract a zip/tar file."""
        fname = self.download(name) #.\data\fra-eng.zip 
        base_dir = os.path.dirname(fname)  # .\data
        data_dir, ext = os.path.splitext(fname)
        if ext == '.zip':
            fp = zipfile.ZipFile(fname, 'r')
        elif ext in ('.tar', '.gz'):
            fp = tarfile.open(fname, 'r')
        else:
            assert False, 'Only zip/tar files can be extracted'
        fp.extractall(base_dir)
        fp.close()
        return os.path.join(base_dir, folder) if folder else data_dir
    
    def read_data_nmt(self):
        """Load the English-French dataset."""
        data_dir = self.download_extract('fra-eng')
        with open(os.path.join(data_dir, 'fra.txt'), 'r', encoding='utf-8') as f:
            return f.read()
    
    def preprocess_nmt(self, text):
        """Preprocess the English-French dataset."""
        def no_space(char, prev_char):
            # Whether it is a punctuation mark and not preceded by a space
            return char in set(',.!?') and prev_char != ' '
        # Replace uppercase letters with lowercase letters 
        # and non-breaking spaces with spaces
        text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
        # Insert space between words and punctuation marks
        out = [' ' + char if i > 0 and no_space(char, text[i-1]) else char
                for i, char in enumerate(text)]
        return ''.join(out)

    def tokenize_nmt(self, text, num_examples = None):
        """Tokenize the English-French dataset"""
        source, target = [], []
        for i, line in enumerate(text.split('\n')):
            if num_examples and i > num_examples:
                break
            parts = line.split('\t')
            if len(parts) == 2:
                source.append(parts[0].split(' '))
                target.append(parts[1].split(' '))
        return source, target
    
    def truncate_pad(self, line, num_steps, padding_token):
        """Truncate or pad sequences"""
        if len(line) > num_steps:
            return line[:num_steps]
        return line + [padding_token] * (num_steps - len(line))
        
    def build_array_nmt(self, lines, vocab, num_steps):
        """Transform text sequences of machine translation into minibatches"""
        lines = [vocab[l] for l in lines]
        lines = [l + [vocab['<eos>']] for l in lines]
        array = torch.tensor([self.truncate_pad(
            l, num_steps, vocab['<pad>']) for l in lines])
        valid_lens = (array != vocab['<pad>']).type(torch.int32).sum(1)
        return array, valid_lens
    
    def load_array(self, data_arrays, batch_size, is_train = True):
        """Construct a PyTorch data iterator"""
        dataset = data.TensorDataset(*data_arrays) 
        # the function of * is to expend it to several positonal parameters
        return data.DataLoader(dataset, batch_size, shuffle=is_train)
        
    def load_data_nmt(self, batch_size, num_steps, num_examples=600):
        """Return the iterator and the vocabularies of the translation dataset"""
        text = self.preprocess_nmt(self.read_data_nmt())
        source, target = self.tokenize_nmt(text, num_examples)
        src_vocab = Vocab(source, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
        target_vocab = Vocab(target, min_freq=2, reserved_tokens=['<pad>', '<bos>', '<eos>'])
        src_array, src_valid_len = self.build_array_nmt(source, src_vocab, num_steps)
        tgt_array, tgt_valid_len = self.build_array_nmt(target, target_vocab, num_steps)
        data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
        data_iter = self.load_array(data_arrays, batch_size)
        return data_iter, src_vocab, target_vocab

class Vocab():
    """Vocabulary for text."""
    def __init__(self, tokens = None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reversed_tokens = []
        # Sort according to frequencies
        counter = self.count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x:x[1],
                                   reverse=True)
        # The index for the unknown token is 0
        # reserved_tokens like '<pad>','<bos>','<>'
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token:idx for idx, token in enumerate(self.idx_to_token)}
        for token, frq in self._token_freqs:
            if frq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __len__(self):
        return len(self.idx_to_token)
    
    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]
    
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
        
    @property
    def unk(self):
        return 0
    
    @property
    def token_freqs(self):
        return self._token_freqs
        
    def count_corpus(self, tokens):
        """Count token frequencies"""
        # Here `tokens` is a 1D list or 2D list
        if len(tokens) == 0 or isinstance(tokens[0], list):
            # Flatten a list of token lists into a list of tokens
            tokens = [token for line in tokens for token in line]
        return collections.Counter(tokens)

def use_svg_display():
    """Use the svg format to display a plot in Jupyter.

    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats('svg')

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

class Animator:
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        """Defined in :numref:`sec_softmax_scratch`"""
        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
        # Ensure figure is shown when running as a script/terminal
        try:
            # Non-blocking show + short pause works for GUI backends
            plt.show(block=False)
            plt.pause(0.001)
        except Exception:
            # As fallback, save a snapshot file (optional)
            try:
                self.fig.savefig('animator_snapshot.png')
            except Exception:
                pass

class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
def grad_clipping(net, theta):
    """Clip the gradient.
    Defined in :numref:`sec_rnn_scratch`"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def sequence_mask(X, valid_lens, value = 0):
    """Mask irrelevant entries in sequences"""
    maxlen = X.size(1)
    # use broadcasting to compare
    mask = torch.arange((maxlen), dtype=torch.float32, 
                        device = X.device)[None, :] < valid_lens[:, None]
    # mask shape(batch_size, max_len) or (batch_size*num of queries, max_len)
    X[~mask] = value
    return X 

def masked_softmax(X, valid_lens):
    """Perform sofrmax operation by masking elements on the last axis."""
    if valid_lens is None:
        return nn.functional.softmax(X, dim = -1)
    else:
        shape = X.shape
        #valid_lens shape:(batch_size，) or (batch_size，num of queries)
        if valid_lens.dim() == 1:
            # reshaped valid_lens to correspond to the reshaped scores correctly
            valid_lens = torch.repeat_interleave(valid_lens, X.shape[1]) #(batch_size * num of queries,)
        else:
            valid_lens = valid_lens.reshape(-1)
        # replace with a very large negative value, whose softmax outputs 0
        X = sequence_mask(X.reshape(-1, X.shape[-1]), valid_lens, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim = -1)
    
class MaskSoftmaxCELoss(nn.CrossEntropyLoss):
    """The softmax cross-entropy loss with masks"""
    # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
    # `label` shape: (`batch_size`, `num_steps`)
    # `valid_len` shape: (`batch_size`,)
    def forward(self, pred, label, label_valid_lens):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, label_valid_lens)
        self.reduction = 'none'
        unweighter_loss = super(MaskSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label) 
        # CrossEntropyLoss input shaped needed (N, C, ...) C means categories
        # self.reduction = 'none'， return unweighter_loss shape(batch_size, num_steps)
        weights_loss = (unweighter_loss * weights).mean(dim = 1) #shape(batch_size)
        return weights_loss

def bleu(pred_seq, label_seq, k):
    """Compute the BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1-len_label/len_pred))
    for n in range(1, k+1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i:i+n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i:i+n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i:i+n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

def show_heatmaps(matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5),
                  cmap='Reds'):
    """Show heatmaps of matrices.

    Defined in :numref:`sec_attention-cues`"""
    use_svg_display()
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize,
                                 sharex=True, sharey=True, squeeze=False)
    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(np.numpy(matrix), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])
    fig.colorbar(pcm, ax=axes, shrink=0.6);

class AdditiveAttention(nn.Module):
    """Additive attention"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs ):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_k(queries), self.W_q(keys)
        #after expansion, queries shape(batch_size, num of quries, num_hiddens)
        #keys shape (batch_size, num of key-value pairs, num_hiddens)
        # use broadcasting for summation
        feature = queries.unqueeze(2) + keys.unqueeze(1)
        feature = torch.tanh(feature)
        scores = self.W_v(feature).squeeze(-1)
        # scores shape (batch_size, num of quries, num of keys)
        # num of key-value pairs == num_steps
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values) #shape (batch_size, num of quries, value dim)
    
class DotProductAttention(nn.Module):
    """Scaled dot product attention"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens = None):
        # Shape of `queries`: (`batch_size`, no. of queries, `d`)
        # Shape of `keys`: (`batch_size`, no. of key-value pairs, `d`)
        # Shape of `values`: (`batch_size`, no. of key-value pairs, value dimension)
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values) #shape (batch_size, num of quries, value dim)

class MultiHeadAttention(nn.Module):
    """Multi-head attention"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias)

    def forward(self, queries, keys, values, valid_lens):
        queries = self.transpose_qkv(self.W_q(queries), self.num_heads)
        keys = self.transpose_qkv(self.W_k(keys), self.num_heads)
        values = self.transpose_qkv(self.W_v(values), self.num_heads)
        # (batch_size*num_heads, no. of queries or key-value pairs, num_hiddens/num_heads)
        if valid_lens is not None:
            # repeat num_heads times in dim 0 to correspond to the q,k,v in dim 0
            valid_lens = torch.repeat_interleave(valid_lens,
                                                 repeats=self.num_heads, dim=0)
        output = self.attention(queries, keys, values, valid_lens) 
        # output (`batch_size` * `num_heads`, no. of queries,num_hiddens/num_heads)
        output_concat = self.transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

    def transpose_qkv(self, X, num_heads):
        """Transposition for parallel computation of multiple attention heads"""
        # input X shape(batch_size, num of queries or k-v pairs, num_hiddens)
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, X.shape[2], X.shape[3]) # shape(batch_size*num_heads, num of q or k-v pairs, num_hiddens/num_heads)

    def transpose_output(self, X, num_heads):
        """Reverse the operation of `transpose_qkv`"""
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)

class PositionalEncoding(nn.Module):
    """Positional encoding"""
    def __init__(self, num_hiddens, dropout, max_len = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        # use broadcasting X shape(maxlen, num_hiddens/2)
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype = torch.float32) / num_hiddens) 
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        # X shape (batch_size, num_steps, num_hiddens)
        return self.dropout(X)
    
class PositionWiseFFN(nn.Module):
    """Positionwise feed-forward network"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
    
class AddNorm(nn.Module):
    """Residual concatenation and layer normalization"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class EncoderBlock(nn.Module):
    """Transformer encoder block"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(key_size, query_size, value_size,
                                            num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
    
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias = False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block"+str(i), 
                        EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                        norm_shape, ffn_num_input, ffn_num_hiddens,
                                        num_heads, dropout, use_bias))
    
    def forward(self, X, valid_lens, *args):
        # scale the embedding(X) by multiply the square of num_hiddens
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens)) 
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[i] = blk.attention.attention.attention_weights
        return X

class DecoderBlock(nn.Module):
    """Transformer decoder block"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, 
                                   num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)
    
    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all target tokens are processed in parallel,
        # so state[2][self.i] is initialized as None.
        # During inference, target tokens are generated step by step,
        # so state[2][self.i] stores the decoded representations up to step i.
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # dec_valid_lens shape (batch_size, num_steps)
            # every row [1, 2,..., num_steps]
            # casual mask
            dec_valid_lens = torch.arange(
                1, num_steps+1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # enc_outputs shape (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state    

class TransformerDecoder(nn.Module):
    def __init__(self,vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(self.num_layers):
            self.blks.add_module("block"+str(i),
                    DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                norm_shape, ffn_num_input, ffn_num_hiddens,
                                num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None]*self.num_layers]
        # [None]*self.num_layers = [None, None,...,None], self.num_layers counts
    
    def forward(self, X, state):
        # if X is 2D(when predict), self.embedding will expend it to 3D (1, 1, num_hiddens)
        X = self.pos_encoding(self.embedding(X)*math.sqrt(self.num_hiddens))
        self._attention_weights = [[None]*len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights

class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, enc_valid_lens):
        enc_outputs = self.encoder(enc_X, enc_valid_lens)
        dec_state = self.decoder.init_state(enc_outputs, enc_valid_lens)
        return self.decoder(dec_X, dec_state)

def train_seq2seq(net, data_iter, lr, num_epochs, tgt_vocab, device):
    """Train a model for sequence to sequence."""
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)
    loss = MaskSoftmaxCELoss()
    net.train()
    animator = Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs])
    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2) # sum of training loss, num of tokens
        for batch in data_iter:
            optimizer.zero_grad()
            X, X_valid_lens, Y, Y_valid_lens = [x.to(device) for x in batch]
            # X and Y shape(batch_size, num_steps), valid_lens (batch_size)
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1) # Teacher forcing
            Y_hat, _ = net(X, dec_input, X_valid_lens)
            l = loss(Y_hat, Y, Y_valid_lens)
            l.sum().backward()
            grad_clipping(net, 1) # lip the gradient
            num_tokens = Y_valid_lens.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)
        if (epoch + 1) % 10 == 0:
            # animator.add(epoch + 1, (metric[0] / metric[1],)) # show image
            print(f'{epoch} loss:{metric[0] / metric[1]:.3f}')
    print(f'loss {metric[0] / metric[1]:.3f}, {metric[1] / timer.stop():.1f} '
          f'tokens/sec on {str(device)}')

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """Predict for sequence to sequence"""
    net.eval()
    dataloader = DataLoad()
    src_tokens = src_vocab[dataloader.preprocess_nmt(src_sentence).split(' ')] + [src_vocab['<eos>']]
    enc_valid_lens = torch.tensor([len(src_tokens)], device=device)
    src_tokens = dataloader.truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = torch.unsqueeze(torch.tensor(src_tokens, dtype = torch.long,
                            device = device), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_lens)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_lens)
    # Add batch axis
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long, device=device), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # use the token with the highest prediction likelihood as the input of the decoder next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


if __name__ == "__main__":
    num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
    lr, num_epochs = 0.005, 200
    ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
    key_size, query_size, value_size  = 32, 32, 32
    norm_shape = [32] # predict need shape [1, 32] so difined norm_shape [32]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA_HUB = dict()
    DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
    DATA_HUB['fra-eng'] = (DATA_URL + 'fra-eng.zip', '94646ad1522d915e7b0f9296181140edcf86a4f5')

    dataload = DataLoad(DATA_HUB)
    train_iter, src_vocab, tgt_vocab = dataload.load_data_nmt(batch_size, num_steps)
    encoder = TransformerEncoder(len(src_vocab), key_size, query_size,
                value_size, num_hiddens, norm_shape, ffn_num_input, num_hiddens,
                num_heads, num_layers, dropout)
    decoder = TransformerDecoder(len(tgt_vocab), key_size, query_size,
                value_size, num_hiddens, norm_shape, ffn_num_input, num_hiddens,
                num_heads, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)
    train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)

    engs = ['go .', "i lost .", 'he\'s calm .', 'i\'m home .']
    fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
    for eng, fra in zip(engs, fras):
        translation, dec_attention_weight_seq = predict_seq2seq(
            net, eng, src_vocab, tgt_vocab, num_steps, device, True)
        print(f'{eng} => {translation}, ',
            f'bleu {bleu(translation, fra, k=2):.3f}')
    # raw_text = dataload.read_data_nmt()
    # print(raw_text[:75])
    # text = dataload.preprocess_nmt(raw_text)
    # print(text[:80])
    # source, target = dataload.tokenize_nmt(text)
    # print(source[:6], '\n', target[:6])
    # src_vocab = Vocab(source, min_freq=2, 
    #                   reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # target_vocab = Vocab(target, min_freq=2,
    #                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
    # print(src_vocab.to_tokens(list(range(10))))
    # print(list(src_vocab.token_to_idx.items())[:10])
    # print(src_vocab.idx_to_token[:10])
    # print(dataload.truncate_pad(src_vocab[source[0]], num_steps=10, padding_token=src_vocab['<pad>']))
    # array, valid_len = dataload.build_array_nmt(source, src_vocab, 10)
    # print(valid_len.shape)
    # print(array[0], valid_len[0])
    # train_iter, src_vocab, tgt_vocab = dataload.load_data_nmt(batch_size=2, num_steps=8)
    # for X, X_valid_len, Y, Y_valid_len in train_iter:
    #     print('X:', X.type(torch.int32))
    #     print('X的有效长度:', X_valid_len)
    #     print('Y:', Y.type(torch.int32))
    #     print('Y的有效长度:', Y_valid_len)
    #     print(Y_valid_len.shape)
    #     break