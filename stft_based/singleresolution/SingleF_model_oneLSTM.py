# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 19:35:38 2021

@author: User
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import torchaudio
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import LSTM, BatchNorm1d, Linear, Parameter
from torch import Tensor
import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

EPS = 1e-8

class cortex_separator(nn.Module):
    def __init__(self,
                 nb_bins=770,
                 n_fft=512,
                 hop=256, 
                 hidden_size=1024,
                 lstm_hidden_size=512,
                 num_layers=3,
                 unidirectional=True, 
                 audio_channels=1,
                 c1=3, 
                 c2=15,
                 c3=65,
                 
                 t=15,
                 N1=32,
                 N2=16,
                 N3=8,
                 nb_output_channels=2,
                 nb_channels = 1,
                 causal=True,use_cortex=True):
        
        """
        Args:
            N: Number of filters in autoencoder
            L: Length of the filters (in samples)
            hop: stride of filters in autoencoder
            B: Number of channels in bottleneck 1 × 1-conv block
            num_layers : Number of BLSTM layers
            

        """
        super(cortex_separator, self).__init__()
        # hyper-parameter
        self.use_cortex = use_cortex
        self.n_fft, self.nb_bins, self.hop = n_fft, nb_bins, hop
        self.nb_output_channels = nb_output_channels
        self.nb_channels = nb_channels
        self.hidden_size, self.num_layers = hidden_size, num_layers
        # self.bidirectional = bidirectional
        self.c1, self.c2, self.c3, self.t, self.N1,self.N2,self.N3= c1, c2, c3, t, N1,N2,N3
       
        
        # Components
        #self.encoder = Encoder_TorchSTFT(n_fft_short, n_fft_long, hop, audio_channels)
        self.cortex = cortex(nb_bins, hidden_size, c1, c2, c3, t, N1, N2, N3, n_fft,causal)
        self.separator = Separator(nb_bins=nb_bins, hidden_size=hidden_size,lstm_hidden_size=lstm_hidden_size, 
                                   nb_layers=num_layers,nb_output_channels=nb_output_channels,nb_channels=nb_channels,
                                   unidirectional=unidirectional,use_cortex=self.use_cortex)    
        
        #self.decoder = Decoder_TorchISTFT(n_fft_short, n_fft_long, hop, audio_channels)


    def forward(self, mag_x):
        """
        Args:
            mixture: [B, c,K, L]
            mixture_lengths: [B]
        Returns:
            est_source: [B, nspk, K, L]
        """
        #mag_x, stft = self.encoder(mixture)
        if self.use_cortex == True:
            cortexfeature = self.cortex(mag_x)
            est_mask,(hn,cn) = self.separator(cortexfeature,mag_x) 
        else:
            # print('no cortex')
            est_mask,(hn,cn) = self.separator(x=mag_x,encoder_mag=mag_x) 
        # est_source = self.decoder(est_mask)
        return est_mask,(hn,cn)




class Encoder_TorchSTFT(nn.Module):
    """Multichannel Short-Time-Fourier Forward transform
    uses hard coded hann_window.
    Args:
        n_fft (int, optional): transform FFT size. Defaults to 4096.
        n_hop (int, optional): transform hop size. Defaults to 1024.
        center (bool, optional): If True, the signals first window is
            zero padded. Centering is required for a perfect
            reconstruction of the signal. However, during training
            of spectrogram models, it can safely turned off.
            Defaults to `true`
        window (nn.Parameter, optional): window function
    """
    def __init__(self, n_fft, n_hop, center=False, window=None):
        super(Encoder_TorchSTFT, self).__init__()

        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center
        self.window = nn.Parameter(torch.hamming_window(n_fft), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """STFT forward path
        Args:
            x (Tensor): audio waveform of
                shape (nb_samples, nb_channels, nb_timesteps)
        Returns:
            STFT (Tensor): complex stft of
                shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
                last axis is stacked real and imaginary
        """

        shape = x.size()     #[batch, 1, 16000]
        #print('shape.input:'+str(x.shape))

        # nb_samples, nb_channels, nb_timesteps = shape

        # pack batch
        x = x.view(-1, shape[-1])   #[batch, 16000]
        #print('shape.pack_batch:'+str(x.shape))

        stft_f = torch.stft(x, n_fft=self.n_fft, hop_length=self.n_hop, window=self.window, center=self.center,\
                                        normalized=False, onesided=True, pad_mode="reflect", return_complex=True)
        
        #return complex [batch,bin,frame]
        
                                        
        # print(stft_f.shape) #([60, 513, 63, 2])

        
        #print('shape.encoder_stft_short:'+str(complex_stft_short.shape))  #[batch, 33, 499] 

        stft_f = torch.view_as_real(stft_f)  #abs,phase值 [batch, 33, 499, 2]
        
        # print('shape.abs:'+str(stft_f.shape)) 
        

        stft_f = stft_f.view(shape[:-1] + stft_f.shape[-3:])  #[batch, 1, 33, 499, 2]


        return stft_f
        
    
class Decoder_TorchISTFT(nn.Module):
    """Multichannel Inverse-Short-Time-Fourier functional
    wrapper for torch.istft to support batches
    Args:
        STFT (Tensor): complex stft of
            shape (nb_samples, nb_channels, nb_bins, nb_frames, complex=2)
            last axis is stacked real and imaginary
        n_fft (int, optional): transform FFT size. Defaults to 4096.
        n_hop (int, optional): transform hop size. Defaults to 1024.
        window (callable, optional): window function
        center (bool, optional): If True, the signals first window is
            zero padded. Centering is required for a perfect
            reconstruction of the signal. However, during training
            of spectrogram models, it can safely turned off.
            Defaults to `true`
        length (int, optional): audio signal length to crop the signal
    Returns:
        x (Tensor): audio waveform of
            shape (nb_samples, nb_channels, nb_timesteps)
    """

    def __init__(self,n_fft: int = 64, n_hop: int = 32, center: bool = False, sample_rate: float = 16000.0,window=None):

        super(Decoder_TorchISTFT, self).__init__()

        self.n_fft= n_fft
        self.n_hop = n_hop
        self.center = center
        self.sample_rate = sample_rate
        self.window = nn.Parameter(torch.hamming_window(n_fft), requires_grad=False)


    def forward(self, X: Tensor, length: Optional[int] = None) -> Tensor:
        shape = X.size()
        X = X.reshape(-1, shape[-3], shape[-2], shape[-1])
        
        # print('shape.mixture_decode:'+str(X.shape))
        
        y = torch.istft(
            torch.view_as_complex(X),
            n_fft=self.n_fft,
            hop_length=self.n_hop,
            window=self.window,
            center=self.center,
            normalized=False,
            onesided=True,
            length=length,
        )
        
        #print('shape.mixture_decode_istft:'+str(y.shape))
        
        y = y.reshape(shape[:-3] + y.shape[-1:])
        
        
        #print('shape.output:'+str(y.shape))

        return y

class cortex(nn.Module):
    def __init__(self, nb_bins, hidden_size, c1, c2, c3, t, N1, N2, N3,n_fft,causal):
        """
        Args:
            N: Number of filters in autoencoder
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """
        super(cortex, self).__init__()
        # Hyper-parameter
        self.causal = causal
        self.n_fft = n_fft
        self.nb_bins = nb_bins
        self.hidden_size = hidden_size
        self.c1 = c1  #3
        self.c2 = c2  #15
        self.c3 = c3  #65
        
        
       
        self.t = t  #15
        self.N1 = N1
        self.N2 = N2
        self.N3 = N3
        self.total = N1+N2
        
        if self.causal == True:
            self.conv_c1 = nn.Conv2d(1, N1, (c1, t) , stride=1, padding = ((self.c1-1)//2,self.t-1))
            self.conv_c2 = nn.Conv2d(1, N2, (c2, t) , stride=1, padding = ((self.c2-1)//2,self.t-1))
            self.conv_c3 = nn.Conv2d(1, N3, (c3, t) , stride=1, padding = ((self.c3-1)//2,self.t-1))
        else:
            self.conv_c1 = nn.Conv2d(1, N1, (c1, t) , stride=1, padding = ((self.c1-1)//2,(self.t-1)//2))
            self.conv_c2 = nn.Conv2d(1, N2, (c2, t) , stride=1, padding = ((self.c2-1)//2,(self.t-1)//2))
            self.conv_c3 = nn.Conv2d(1, N3, (c3, t) , stride=1, padding = ((self.c3-1)//2,(self.t-1)//2))
        
        self.fc2 = Linear(self.nb_bins * self.total, self.nb_bins, bias=False)

        self.fc2_ = Linear(self.nb_bins * 2, self.nb_bins, bias=False)
        
        
    def forward(self, x):
        """
        Args:
            x: [batch, 1, 770, 63]  [batch, channel , bins, frames]
        Returns:
            [frame,batch,bins]
        """
            
        batch, channel,  bins, frames = x.data.shape 
        
        
     
        x_ = x.permute(3, 0, 1, 2)
        
        # print('start==========================')
        clusters1 = self.conv_c1(x) 
        # print('shape.2dconv_c1:', clusters1.shape)  #[batch, channel , bins, frames]
        clusters2 = self.conv_c2(x)
        # print('shape.2dconv_c2:', clusters2.shape)  #[batch, channel , bins, frames]
        # clusters3 = self.conv_c3(x_short)
        # print('shape.2dconv_c3:', cluster3.shape)  #([2, 6, 257, 63])
        
        
        if self.causal==True:
            clusters1 = clusters1[:, :,:, :-(self.t-1)].contiguous()
            # print('shape.2dconv_c1:', clusters1.shape)
            clusters2 = clusters2[:, :,:, :-(self.t-1)].contiguous()
            # print('shape.2dconv_c2:', clusters2.shape)

        
        C = torch.cat([clusters1, clusters2],dim=1)
        C = C.permute(3, 0, 1, 2) 
        nb_frames, nb_samples, channels, short_bins = C.data.shape 
        
        C = self.fc2(C.reshape(-1, (channels * self.nb_bins))) 
        C = C.reshape(nb_frames, nb_samples, 1 , self.nb_bins)

        
        #skip connection
        C2 = torch.cat([C,x_],dim=2)
        C2 = self.fc2_(C2.reshape(-1, (2 * self.nb_bins)))
        C2 = C2.reshape(nb_frames, nb_samples, 1 , self.nb_bins)
        C2 = C2.permute(1,2,3,0) #batch channel bin frame
        mixture_C = C2
        mixture_C = F.relu(mixture_C)
        # print('shape.RELU:', mixture_C.shape)  #([2, 63, 512])
        
        return mixture_C
        
        
        
        
class Separator(nn.Module):
    """OpenUnmix Core spectrogram based separation module.

    Args:
        nb_bins (int): Number of input time-frequency bins (Default: `4096`).
        nb_channels (int): Number of input audio channels (Default: `2`).
        hidden_size (int): Size for bottleneck layers (Default: `512`).
        nb_layers (int): Number of Bi-LSTM layers (Default: `3`).
        unidirectional (bool): Use causal model useful for realtime purpose.
            (Default `False`)
        input_mean (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to zeros(nb_bins)
        input_scale (ndarray or None): global data mean of shape `(nb_bins, )`.
            Defaults to ones(nb_bins)
        max_bin (int or None): Internal frequency bin threshold to
            reduce high frequency content. Defaults to `None` which results
            in `nb_bins`
    """

    def __init__(   
        self,
        nb_bins=770,
        hidden_size=1024,
        lstm_hidden_size=512,
        nb_layers=3,
        nb_output_channels=2,
        nb_channels=1,
        unidirectional=True,
        input_mean=None,
        input_scale=None,
        max_bin=None,use_cortex =True
    ):
        super(Separator, self).__init__()
           
        self.nb_bins = nb_bins
        self.nb_output_channels = nb_output_channels
        self.nb_output_bins = nb_bins
        self.nb_channels = nb_channels
        self.hidden_size = hidden_size
        self.use_cortex = use_cortex
        self.unidirectional = unidirectional
        self.lstm_hidden_size = lstm_hidden_size

        
   
        self.lstm = LSTM(
            input_size = hidden_size,
            hidden_size = self.lstm_hidden_size,
            num_layers = nb_layers,
            bidirectional = not self.unidirectional,
            batch_first = False,
            dropout=0.4 if nb_layers > 1 else 0,
        )
        
        
        self.fc_ini = Linear(self.nb_bins , hidden_size, bias=False)
        self.bn_ini = BatchNorm1d(hidden_size)
        
        self.fc = Linear(lstm_hidden_size+hidden_size , hidden_size, bias=False)
        self.bn = BatchNorm1d(hidden_size)
        
     
        
        self.fc2 = Linear(in_features = hidden_size, out_features = nb_bins * self.nb_output_channels, bias=False)
        self.bn2 = BatchNorm1d(nb_bins * self.nb_output_channels)
        

        

    def forward(self, x: Tensor, encoder_mag : Tensor,hn: Optional[int] = None,cn: Optional[int] = None) -> Tensor:
        """
        Args:
            x: frame,nb_samples,bins
            encoder_mag : nb_samples, nb_channels, nb_bin, nb_frames
            

        Returns:
            Tensor: filtered spectrogram of shape
                `(nb_samples, nb_channels, nb_bins, nb_frames)
        """

        # permute so that batch is last for lstm

        # get current spectrogram shape
        nb_samples, nb_channels, nb_bin, nb_frames = encoder_mag.data.shape

        #x.shape [sample frame hiddensize]
        
        x = x.permute(3,0,1,2)
        
        x = self.fc_ini(x.reshape(-1, self.nb_bins)) #([126, 512])  
        x = self.bn_ini(x)
        x = x.reshape(nb_frames, nb_samples, self.hidden_size) 
        x = F.relu(x)
        # print(x.shape)

       
        if hn==None:
            hn=torch.zeros([self.lstm.num_layers,nb_samples, self.lstm.hidden_size]).to(device)
            cn=torch.zeros([self.lstm.num_layers,nb_samples, self.lstm.hidden_size]).to(device)
      
        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x,(hn,cn))
        # print('shape.LSTM_output:', len(vocal_lstm_out)) #

        # lstm skip connection
        x = torch.cat([x, lstm_out[0]], -1)
        # print('shape.skip connection_mask_v:', x.shape) #([10, 63, 1024])


        # first dense stage + batch norm
        x = self.fc(x.reshape(-1, x.shape[-1]))
        x = self.bn(x)
        x = F.relu(x) 
        # print('shape.fc_x:', x.shape) #([630, 512])

        
        x = self.fc2(x)
        x = self.bn2(x)
        
        
        x = x.reshape(nb_frames, nb_samples, nb_bin*self.nb_output_channels)

        mask = x.permute(1,2,0)  


        
        # print(mask.shape)
        mask = mask.view(nb_samples, self.nb_output_channels, nb_bin, nb_frames) 
        # print(mask.shape)
        # print(mask_v.shape)

        # since our output is non-negative, we can apply RELU
        
        # print('mix_shape:'+str(encoder_mag.shape))  #([10, 1, 257, 63])
        est_mask = F.relu(mask) * encoder_mag
        # est_music = F.relu(mask_m) * encoder_mag
        
        
        # permute back to (nb_samples, nb_channels, nb_bins, nb_frames)
        return est_mask,lstm_out[1]