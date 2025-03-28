"""
Custom RelTR to pretrain object detection part
"""

import copy
import torch
from typing import Optional
from torch import nn, Tensor
from .transformer import TransformerEncoder, TransformerEncoderLayer, _get_activation_fn


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                activation='relu', normalize_before=False, return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                 activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, 
                                                dropout, activation)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, 
                                          return_intermediate=return_intermediate_dec)
        
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, entity_embed, pos_embed):
         
        # flatten NxCxHxW to HxWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)

        entity_embed, entity = torch.split(entity_embed, c, dim=1)
        entity_embed = entity_embed.unsqueeze(1).repeat(1, bs, 1)
        entity = entity.unsqueeze(1).repeat(1, bs, 1)

        mask = mask.flatten(1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(entity, memory, memory_key_padding_mask=mask, pos=pos_embed, entity_pos=entity_embed)
        return hs.transpose(1, 2)




class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate

    
    def forward(self, entity, memory, entity_pos: Optional[Tensor] = None, tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,):
        output_entity = entity
        intermediate_entity = []
        
        for layer in self.layers:
            output_entity = layer(output_entity, entity_pos, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_mask=memory_mask, memory_key_padding_mask=memory_key_padding_mask, pos=pos)
            
            if self.return_intermediate:
                intermediate_entity.append(output_entity)

        if self.return_intermediate:
            return torch.stack(intermediate_entity)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super().__init__()
        self.activation = _get_activation_fn(activation)

        # entity part
        self.self_attn_entity = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout2_entity = nn.Dropout(dropout)
        self.norm2_entity = nn.LayerNorm(d_model)

        self.cross_attn_entity = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1_entity = nn.Dropout(dropout)
        self.norm1_entity = nn.LayerNorm(d_model)

        # ffn
        self.linear1_entity = nn.Linear(d_model, dim_feedforward)
        self.dropout3_entity = nn.Dropout(dropout)
        self.linear2_entity = nn.Linear(dim_feedforward, d_model)
        self.dropout4_entity = nn.Dropout(dropout)
        self.norm3_entity = nn.LayerNorm(d_model)

    def forward_ffn_entity(self, tgt):
        tgt2 = self.linear2_entity(self.dropout3_entity(self.activation(self.linear1_entity(tgt))))
        tgt = tgt + self.dropout4_entity(tgt2)
        tgt = self.norm3_entity(tgt)
        return tgt

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


    def forward(self, tgt_entity, entity_pos, memory, tgt_mask: Optional[Tensor] = None, 
                tgt_key_padding_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] =  None,
                memory_key_padding_mask : Optional[Tensor] = None, pos: Optional[Tensor] = None):

        # entity layer
        q_entity = k_entity = self.with_pos_embed(tgt_entity, entity_pos)
        tgt2_entity = self.self_attn_entity(q_entity, k_entity, value=tgt_entity, attn_mask=tgt_mask,
                                            key_padding_mask=tgt_key_padding_mask)[0]
        tgt_entity = tgt_entity + self.dropout2_entity(tgt2_entity)
        tgt_entity = self.norm2_entity(tgt_entity)

        tgt2_entity = self.cross_attn_entity(query=self.with_pos_embed(tgt_entity, entity_pos),
                                             key=self.with_pos_embed(memory, pos), value=memory, attn_mask=memory_mask,
                                             key_padding_mask=memory_key_padding_mask)[0]
        tgt_entity = tgt_entity + self.dropout1_entity(tgt2_entity)
        tgt_entity = self.norm1_entity(tgt_entity)
        tgt_entity = self.forward_ffn_entity(tgt_entity)
        return tgt_entity
    

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_custom_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )