import torch
import torch.nn as nn
from modules import ConvBNRelu, ConvSigm
from modules import InceptionModule, InceptionImmediate


class InceptionEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            InceptionModule(in_channels, 1),
            InceptionModule(1, 1),
            ConvBNRelu(1, 24, (1,1)),
            InceptionModule(24, 24),
            InceptionModule(24, 24),
            ConvBNRelu(24, out_channels, (1,1)),
        )

    def forward(self, secret):
        code = self.encoder(secret)
        return code # [-1, 48, 32, 32]


class InceptionEmbedder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionEmbedder, self).__init__()

        # Embedder
        self.incep1 = InceptionImmediate(in_channels, 3)
        self.incep2 = InceptionModule(6, 6)
        self.conv = ConvSigm(6, out_channels, (1,1))
        # use 'sigmoid' instead of 'relu' to limit output in range [0,1]

    def forward(self, code, image):
        # build embedder
        code_b1, code_b2, code_conv = self.incep1(code)
        concat = torch.cat((code_conv, image), dim=1) # [-1, 6, 128, 128]
        incep = self.incep2(concat)
        hidden = self.conv(incep)

        # immediate process hidden
        hidden_b1, hidden_b2, hidden_conv = self.incep1(hidden)
        return (code_b1, code_b2, hidden_b1, hidden_b2, hidden) # [-1, 3, 128, 128]


class InvarianceLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InvarianceLayer, self).__init__()
        
        # set activation='tanh', use_bias=False to follow the paper setting
        self.dense_n = nn.Linear(in_channels, out_channels, bias=False)
        self.tanh_activation = nn.Tanh()

    def forward(self, image): # [b,3,h,w]
        # build invariance layer
        image = image.permute(0, 2, 3, 1) # [b,h,w,3]
        info_n = self.dense_n(image) # [b,h,w,n]
        info_n = info_n.permute(0, 3, 1, 2) # [b,n,h,w]
        # debug: <yes>
        info_n = self.tanh_activation(info_n)
        return info_n


class InceptionExtractor(nn.Module):
    def __init__(self, n, in_channels, out_channels):
        super(InceptionExtractor, self).__init__()

        # extractor
        self.extractor = nn.Sequential(
            InceptionModule(in_channels, n),
            InceptionModule(n, n),
            ConvBNRelu(n, 3, (1,1))
        )

    def forward(self, info_n):
        extracted_image = self.extractor(info_n)
        return extracted_image


class InceptionDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionDecoder, self).__init__()

        # decoder
        self.decoder = nn.Sequential(
            InceptionModule(in_channels, 48),
            InceptionModule(48, 48),
            ConvBNRelu(48, 24, (1,1)),
            InceptionModule(24, 24),
            InceptionModule(24, 24),
            ConvSigm(24, 1, (1,1))
        )
        # use 'sigmoid' instead of 'relu' to limit output in range [0,1]

    def forward(self, code):
        decoded_secret = self.decoder(code)
        return decoded_secret


class Pure(nn.Module):
    def __init__(self, image_size, en_channels, em_channels, secret_size, n, inva_channels, ex_channels, de_channels):
        super(Pure, self).__init__()
        # image [height, width]
        self.image_h, self.image_w = image_size
        # secret [height, width]
        self.secret_h, self.secret_w = secret_size
        
        # encoder-embedder
        self.encoder = InceptionEncoder(en_channels[0], en_channels[1])
        self.embedder = InceptionEmbedder(em_channels[0], em_channels[1])
        
        # invariance-extractor-decoder
        self.invariance = InvarianceLayer(inva_channels[0], inva_channels[1])
        self.extractor = InceptionExtractor(n, ex_channels[0], ex_channels[1])
        self.decoder = InceptionDecoder(de_channels[0], de_channels[1])
    
    def forward(self, image, secret):
        # encoder, embedder
        code = self.encoder(secret) # [b,48,32,32]
        code = torch.reshape(code, (code.size(0), 3, self.image_h, self.image_w)) # [b,3,128,128]
        code_b1, code_b2, hidden_b1, hidden_b2, hidden = self.embedder(code, image)
        
        # invariance layer, extractor and decoder
        info_n = self.invariance(hidden)
        extracted_code = self.extractor(info_n) # [b,3,128,128]
        reshaped_extracted_code = torch.reshape(extracted_code, (extracted_code.size(0), 48, self.secret_h, self.secret_w)) # [b,48,32,32]
        decoded_secret = self.decoder(reshaped_extracted_code) # [b,1,32,32]
        
        return (code, code_b1, code_b2, hidden, hidden_b1, hidden_b2, info_n, extracted_code, decoded_secret)
        