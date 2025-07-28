import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from spectralgpt.spec_gpt_model import mae_vit_base_patch8_96
from mamba import create_model_mamba, create_model_mamba2
from einops.layers.torch import Rearrange

def load_spectralgpt_model():
    checkpoint_path='spectralgpt/SpectralGPT.pth' 
    download_url='https://zenodo.org/records/13139925/files/SpectralGPT.pth'
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    if not os.path.isfile(checkpoint_path):
        print("Weights file not found. Downloading...")
        os.system(f"wget {download_url} -O {checkpoint_path}")
    
    model = mae_vit_base_patch8_96()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    
    for k in ['pos_embed_spatial', 'pos_embed', 'decoder_pos_embed', 'patch_embed.proj.weight',
              'decoder_pos_embed_spatial', 'patch_embed.proj.bias', 'head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    
    return model

def conv33(inchannel, outchannel):
    return nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1)

import torch
class Unmixing(nn.Module):
    def __init__(self, band_Number, endmember_number, 
                drop_out, col, patch_size, mamba_dim , lamda_1, lamda_2):
        super(Unmixing, self).__init__()
        self.endmember_number = endmember_number
        self.band_number = band_Number
        self.col = col
        self.layer1 = nn.Sequential(
            conv33(band_Number, 96),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(96),
            nn.Dropout(drop_out),
            conv33(96, 48),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(48),
            nn.Dropout(drop_out),
            conv33(48, endmember_number),
        )

        self.lamda_1 = lamda_1
        self.lamda_2 = lamda_2
        self.encodelayer = nn.Sequential(nn.Softmax(dim=1))

        self.decoderlayer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=endmember_number,
                out_channels=band_Number,
                kernel_size=(1, 1),
                bias=False,
            ), )
        
        # mamba init
        dim_mamba = mamba_dim 
        self.vtrans_freq = create_model_mamba2(d_model=dim_mamba, num_blocks=1, patch_size=patch_size, headdim=2)
        h, w = self.col//patch_size, self.col//patch_size
        self.to_img = nn.Sequential(Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=h, w=w),) # Reconstruction

        # spectral gpt 
        self.spectral_gpt = load_spectralgpt_model()
        for param in self.spectral_gpt.parameters():
            param.requires_grad = False
        self.linear_transform = nn.Linear(768, (col**2))
        
    def forward_vision(self, x, abundance):
        
        # frequency Embeddings from Mamba
        # freq_tensor = torch.fft.fft2(x, dim=(-2, -1))
        freq_tensor = x
        cls_emb = self.vtrans_freq(freq_tensor.float().cuda())
        layer1out = self.to_img(cls_emb)
        
        # guided abundance frequency Embeddings using from spectral_gpt
        abundance = F.interpolate(abundance, size=(96, 96), mode='bilinear', align_corners=False)
        spectral_emb, _, _ = self.spectral_gpt.forward_encoder(abundance, 0.5)
        spectral_emb = spectral_emb[:, :self.endmember_number, :]
        spectral_emb = self.linear_transform(spectral_emb)
        spectral_emb = spectral_emb.view(layer1out.shape)
        
        
        return  layer1out, spectral_emb

    
    def forward(self, x, mask, abundance):
        # original
        layer1out = self.layer1(x)
        en_result1 = layer1out / mask
        
        mamba2_emb , gpt_emb = self.forward_vision(en_result1, abundance)
        en_result1 = gpt_emb + layer1out  * self.lamda_1
        en_result1 = en_result1 + mamba2_emb * self.lamda_2
        
        en_result1 = self.encodelayer(en_result1)
        de_result1 = self.decoderlayer4(en_result1)

        return en_result1, de_result1
