import math
import torch
from typing import *
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer

'''
This file contains two transformer-based models specifically designed for multivariate time series (mvts) data.
TransformerEncoder: The model takes as input a batch of mvts data and outputs a batch of encoded mvts data that is then used for the task of autoregressive denoising.
Autoregressive denoising is when the model is trained to fill in data that has been masked out. It is important when operating in the data-sparse regime since it allows the model to warm up its weights by learning the dependencies between the different features in the data, leading to better performance on downstream tasks.
CNNModel: simple CNN that processes mvts for the purpose of binary classification and Grad-CAM.
CombinedModel: combines the transformer with the CNN for classification. The transformer creates embeddings of the mvts of the same dimension and then feeds this into the CNN for classification. Because the embeddings retain their positional information, the saliency maps from Grad-CAM can be derived from these embeddings and projected safely back onto the input mvts.
For mvts, one uses learned positional encodings and batch normalization in the transformer encoder layers, hence the additional classes: LearnablePositionalEncoding and TransformerBatchNormEncoderLayer.
'''


class TransformerEncoder(nn.Module):
    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1, freeze=False, batch_norm=False):
        super(TransformerEncoder, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = LearnablePositionalEncoding(d_model, dropout=dropout*(1.0 - freeze), max_len=max_len)
        if batch_norm:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, n_heads, dim_feedforward, dropout*(1.0 - freeze))
        else:
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout*(1.0 - freeze))
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, feat_dim)
        self.act = F.gelu
        self.dropout1 = nn.Dropout(dropout)
        self.feat_dim = feat_dim

    def forward(self, x, padding_masks):
        """
        Args:
            x: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """
        # permute because pytorch convention for transformers is [seq_length, batch_size, feat_dim]. padding_masks [batch_size, feat_dim]
        inp = x.permute(1, 0, 2)
        # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
        inp = self.project_inp(inp) * math.sqrt(self.d_model)
        inp = self.pos_enc(inp)  # add positional encoding

        # NOTE: logic for padding masks is reversed to comply with definition in MultiHeadAttention, TransformerEncoderLayer
        # (seq_length, batch_size, d_model)
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(output)
        output = output.permute(1, 0, 2)  # (batch_size, seq_length, d_model)

        embedding = output.clone()

        output = self.dropout1(output)
        # Most probably defining a Linear(d_model,feat_dim) vectorizes the operation over (seq_length, batch_size).
        # (batch_size, seq_length, feat_dim)
        output = self.output_layer(output)
        return output, embedding
    

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(64 * 35 * 40, 20)
        self.fc2 = nn.Linear(20, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        logits = self.fc2(x)
        return logits
    
class CombinedModel(nn.Module):
    def __init__(self, transformer_model, cnn_model):
        super(CombinedModel, self).__init__()
        self.transformer_model = transformer_model
        self.cnn_model = cnn_model

    def forward(self, x, padding_mask):
        transformer_output, embedding = self.transformer_model(x, padding_mask)
        embedding = embedding.unsqueeze(1)
        logits_output = self.cnn_model(embedding)
        return logits_output
    

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=40):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        # requires_grad automatically set to True
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        """
        Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class TransformerBatchNormEncoderLayer(nn.modules.Module):
    """
    This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """
    def __init__(self, d_model, nhead, dim_feedforward=256, dropout=0.1):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.activation = F.relu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask = None, src_key_padding_mask = None, is_causal = False):
        """
        Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        # src = src.reshape([src.shape[0], -1])  # (batch_size, seq_length * d_model)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # (seq_len, batch_size, d_model)
        src = src.permute(1, 2, 0)  # (batch_size, d_model, seq_len)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)  # restore (seq_len, batch_size, d_model)
        return src