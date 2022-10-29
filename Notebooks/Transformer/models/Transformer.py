import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_val, DataEmbedding_wo_temp, DataEmbedding_wo_pos_val, DataEmbedding_wo_val_temp,\
    DataEmbedding_wo_pos_temp
import numpy as np


class Transformer(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Transformer, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        # if configs.embed_type == 0:
        # self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.positional_embedding,
        #                                    configs.value_embedding, configs.temporal_embedding, configs.embed,
        #                                    configs.freq, configs.dropout)
        # self.dec_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.positional_embedding,
        #                                    configs.value_embedding, configs.temporal_embedding, configs.embed,
        #                                    configs.freq, configs.dropout)
        self.pos = configs.positional_embedding
        self.val = configs.value_embedding
        self.temp = configs.temporal_embedding
        self.enc_embedding = DataEmbedding_wo_val_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.dec_embedding = DataEmbedding_wo_val_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        # if self.pos == "True" and self.val == "True" and self.temp == "True":
        #     print("here 1")
        #     self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        #     self.dec_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        # if self.pos == "True" and self.val == "True" and self.temp == 'False':
        #     print("here 2")
        #     self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        #     self.dec_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        # if self.pos == "True" and self.val == 'False' and self.temp == "True":
        #     print("here 3")
        #     self.enc_embedding = DataEmbedding_wo_val(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout, configs.batch_size)
        #     self.dec_embedding = DataEmbedding_wo_val(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout, configs.batch_size)
        # if self.pos == "True" and self.val == 'False' and self.temp == 'False':
        #     print("here 4")
        #     self.enc_embedding = DataEmbedding_wo_val_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        #     self.dec_embedding = DataEmbedding_wo_val_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        # if self.pos == 'False' and self.val == "True" and self.temp == "True":
        #     print("here 5")
        #     self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        #     self.dec_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        # if self.pos == 'False' and self.val == "True" and self.temp == 'False':
        #     print("here 6")
        #     self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        #     self.dec_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        # if self.pos == 'False' and self.val == 'False' and self.temp == "True":
        #     print("here 7")
        #     self.enc_embedding = DataEmbedding_wo_pos_val(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        #     self.dec_embedding = DataEmbedding_wo_pos_val(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        # if self.pos is False & self.val is False & self.temp is False:
        #     self.enc_embedding = DataEmbedding_wo_pos_val_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        #     self.enc_embedding = DataEmbedding_wo_pos_val_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
                # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor
                                      ,attention_dropout=configs.dropout
                                      # ,output_attention=configs.output_attention
                                      ), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        # if self.output_attention:
        #     return dec_out[:, -self.pred_len:, :], attns
        # else:
        # a = 1
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]
