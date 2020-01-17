import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def make_positional_encoding(max_length, embedding_size):
    import numpy as np
    
    time = np.pi * torch.arange(0, max_length).float()
    freq_dividers = torch.arange(1, embedding_size // 2 + 1).float()
    inputs = time[:, None] / freq_dividers[None, :]
    
    result = torch.zeros(max_length, embedding_size)
    result[:, 0::2] = torch.sin(inputs)
    result[:, 1::2] = torch.cos(inputs)
    return result

class myBertModel(nn.Module):
    def __init__(self, bert_model, padding_idx):
        super().__init__()
        self.bert_model = bert_model
        self.padding_idx = padding_idx

    def forward(self, in_tensor):
        pad_mask = self.get_pad_mask(in_tensor)
        return self.bert_model(in_tensor, attention_mask=pad_mask)[0], pad_mask

    def get_pad_mask(self, in_tensor):
        return (in_tensor != self.padding_idx) * 1.0

    def get_embedding(self):
        return self.bert_model.get_input_embeddings()

def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class myTorchDecoder(nn.Module):
    def __init__(self, emb_layer, d_model=768, nhead=2, dim_feedforward=512, num_layers=2, padding_idx=0):
        super().__init__()
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
            , num_layers=num_layers
        )
        self.padding_idx = padding_idx
        self.embeddings = emb_layer
        self.embedding_size = emb_layer.embedding_dim

    def forward(self, tgt_token_ids, memory, memory_key_padding_mask, max_out_len=None, calc_cross_attention=False):
        if max_out_len is None:
            batch_size, max_out_len = tgt_token_ids.shape
            fact_out_len = max_out_len
        else:
            batch_size, fact_out_len = tgt_token_ids.shape
            
        device = tgt_token_ids.device

        tgt_token_ids = tgt_token_ids.transpose(0,1)
        target_dependency_mask = generate_square_subsequent_mask(fact_out_len).to(device)

        target_padding_mask = (tgt_token_ids == self.padding_idx).transpose(0,1).to(device)
        tgt_embs = self.embeddings(tgt_token_ids)
        tgt_pos_codes = make_positional_encoding(max_out_len, self.embedding_size)[:fact_out_len, :].unsqueeze(1).to(device)
        tgt_embs = tgt_embs + tgt_pos_codes

        memory = memory.transpose(0,1)
        memory_key_padding_mask = (memory_key_padding_mask == 0).to(device)

        target_features = self.decoder(
            tgt_embs
            , memory
            , tgt_mask=target_dependency_mask
            , tgt_key_padding_mask=target_padding_mask
            , memory_key_padding_mask=memory_key_padding_mask
        )

        if calc_cross_attention:
            decoder_layer = self.decoder.layers[0]
            tgt2 = decoder_layer.self_attn(
                tgt_embs, tgt_embs, tgt_embs, attn_mask=target_dependency_mask,key_padding_mask=target_padding_mask
            )[0]
            tgt_embs = tgt_embs + tgt2
            tgt_embs = decoder_layer.norm1(tgt_embs)
            cross_attention = decoder_layer.multihead_attn(
                tgt_embs, memory, memory, attn_mask=None, key_padding_mask=memory_key_padding_mask
            )

            return target_features.transpose(0,1), cross_attention

        return target_features.transpose(0,1)

    def get_pad_mask(self, in_tensor):
        return (in_tensor != self.padding_idx) * 1.0

    def get_embedding_dim(self):
        return self.embeddings.num_embeddings, self.embeddings.embedding_dim

class BertDecoderModel(nn.Module):
    def __init__(self, bert_model, torch_decoder, emb_size_decoder, vocab_size_out):
        super().__init__()
        self.encoder = bert_model
        self.decoder = torch_decoder
        self.linear_out = nn.Linear(emb_size_decoder, vocab_size_out)

    def forward(self, tensor_in, tensor_out, max_out_len=None, return_seq_generation_params=False, encoder_memory=None, encoder_mask=None):
        if (encoder_memory is None) | (encoder_mask is None):
            encoder_memory, encoder_mask = self.encoder(tensor_in)
        decoder_hidden = self.decoder(tensor_out, encoder_memory, encoder_mask, max_out_len)

        logits = self.linear_out(decoder_hidden)

        if return_seq_generation_params:
            return logits, encoder_memory, encoder_mask
        else:
            return logits