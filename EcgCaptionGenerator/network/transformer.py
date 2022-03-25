import torch
from torch import nn
from torch.nn import Module
import math
from EcgCaptionGenerator.network.positional_encoding import PositionalEncoding
#########################################
#               MAIN MODEL              #
#########################################
class TransformerModule(Module):
    def __init__(self, d_model, nhead, num_layers):
        super(TransformerModule, self).__init__()

        self.positional_encoding = PositionalEncoding(d_model)
        self.positional_transformer = PositionalEncoding(d_model, max_len=10)

        self.tgt_mask = None

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers, norm=None)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers, norm=None)

    def forward_one_step(self, image_features, tgt, tgt_key_padding_mask, attended_features=None):
        if attended_features is None:
            attended_features = self.transformer_encoder(image_features) 

        tgt = self.positional_encoding(tgt)
        out = self.transformer_decoder(tgt, attended_features, tgt_key_padding_mask=tgt_key_padding_mask)
        return out, attended_features

    def forward(self, image_features, tgt, tgt_key_padding_mask):
        image_features = self.positional_transformer(image_features)
        attended_features = self.transformer_encoder(image_features)
        
        # attended : (batch, num_features, feature_size)
        
        # tgt = tgt # (seq, batch, embedding)
        tgt = self.positional_encoding(tgt)
        # print(tgt.shape)
        device = tgt.device
        if self.tgt_mask is None or self.tgt_mask.size(0) != len(tgt):
            mask = self._generate_square_subsequent_mask(len(tgt)).to(device)
            self.tgt_mask = mask  # (seq_length, seq_length)
            # print('mask shape', self.tgt_mask.shape)
        # tgt_key_padding_mask : (batchsize, seq_length)
        out = self.transformer_decoder(tgt, attended_features, tgt_mask=self.tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return out # (seq, batch, embed)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask