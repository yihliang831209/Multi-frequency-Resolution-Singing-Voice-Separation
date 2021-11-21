# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 16:32:17 2021

@author: VictoriaChing
"""

from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
# from dataclasses import dataclass, field
import scipy.io as sio
import scipy
import scipy.signal as signal
import torch 
from torch import distributed, nn
import SingleF_model_oneLSTM as wenchen_Net 
# from utils import human_seconds, load_model, save_model, sizeof_fmt
import numpy as np
from torch.autograd import Variable
import timeit
from tensorboardX import SummaryWriter 
import scipy.io.wavfile as wav
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



nb_bins=513
num_layers=1
unidirectional=True 
causal = True
hop = 256
hidden_size = 1024
lstm_hidden_size = 512
nfft = 1024 
use_cortex = False
# nfft_S = 512

batch_size = 30
epochs_size = 80

arfa=0.99

# model = wenchen_Net.cortex_separator(hidden_size=hidden_size,lstm_hidden_size=lstm_hidden_size,
#                                      unidirectional=unidirectional,nb_bins=nb_bins,causal=causal,
#                                      use_cortex=use_cortex,num_layers=num_layers).to(device)
# print(model)

def compute_measures(se,s):
    Rss=s.transpose().dot(s)
    this_s=s

    a=this_s.transpose().dot(se)/Rss
    e_true=a*this_s
    e_res=se-a*this_s
    Sss=np.sum((e_true)**2)
    Snn=np.sum((e_res)**2)

    SDR=10*np.log10(Sss/Snn)

    return SDR



def wavwrite(fn, data, fs):
    data = scipy.array(scipy.around(data * 2**(15)), dtype = "int16")
    wav.write(fn, fs, data)



class Wavedata(Dataset):
    def __init__(self,mix,vocal_music):
        self.mix = mix
        self.vocal_music = vocal_music
    def __len__(self):

        return len(self.mix[:,0,0])

    def __getitem__(self,idx):
        data = self.mix[idx,:,:]
        target = self.vocal_music[idx,:,:]
        
        return torch.tensor(data).float(), torch.tensor(target).float()
    

def scale(x,y,op_based,fs):
    a = x.shape[0]
    temp = int(a/op_based)
    # print(temp)
    end = int(temp*op_based)
    # print(end)
    x = x[0:end,:]
    y = y[:,0:end,:]
    # print(y.shape[1])
    x = np.reshape(x,[-1,fs*op_based])
    y = np.reshape(y, [2,-1,fs*op_based])
    y = y.transpose((1,0,2))
   
    return x,y


        

#____Input training data____#
print('Data loading....')
train_folder = 'D:\yihliang_博班\RTK_separation\DATA\creat_dsd100_data\DSD100_16k_100percent/'
data_1 = sio.loadmat(train_folder+'/DSD100_16k_100percentVocal_pairedMix_randomMix_1.mat')
data_2 = sio.loadmat(train_folder+'/DSD100_16k_100percentVocal_pairedMix_randomMix_2.mat')
data_3 = sio.loadmat(train_folder+'/DSD100_16k_100percentVocal_pairedMix_randomMix_3.mat')
data_4 = sio.loadmat(train_folder+'/DSD100_16k_100percentVocal_pairedMix_randomMix_4.mat')
data_5 = sio.loadmat(train_folder+'/DSD100_16k_100percentVocal_pairedMix_randomMix_5.mat')
data_6 = sio.loadmat(train_folder+'/DSD100_16k_100percentVocal_pairedMix_randomMix_6.mat')
data_7 = sio.loadmat(train_folder+'/DSD100_16k_100percentVocal_pairedMix_randomMix_7.mat')
data_8 = sio.loadmat(train_folder+'/DSD100_16k_100percentVocal_pairedMix_randomMix_8.mat')
data_9 = sio.loadmat(train_folder+'/DSD100_16k_100percentVocal_pairedMix_randomMix_9.mat')
data_10 = sio.loadmat(train_folder+'/DSD100_16k_100percentVocal_pairedMix_randomMix_10.mat')
data_11 = sio.loadmat(train_folder+'/DSD100_16k_100percentVocal_pairedMix_randomMix_11.mat')


x_1_train=data_1['x'][:,:].transpose((1,0))
y_1_train=data_1['y'][:,:,:].transpose((2,1,0))
x_2_train=data_2['x'][:,:].transpose((1,0))
y_2_train=data_2['y'][:,:,:].transpose((2,1,0))
x_3_train=data_3['x'][:,:].transpose((1,0))
y_3_train=data_3['y'][:,:,:].transpose((2,1,0))
x_4_train=data_4['x'][:,:].transpose((1,0))
y_4_train=data_4['y'][:,:,:].transpose((2,1,0))
x_5_train=data_5['x'][:,:].transpose((1,0))
y_5_train=data_5['y'][:,:,:].transpose((2,1,0))
x_6_train=data_6['x'][:,:].transpose((1,0))
y_6_train=data_6['y'][:,:,:].transpose((2,1,0))
x_7_train=data_7['x'][:,:].transpose((1,0))
y_7_train=data_7['y'][:,:,:].transpose((2,1,0))
x_8_train=data_8['x'][:,:].transpose((1,0))
y_8_train=data_8['y'][:,:,:].transpose((2,1,0))
x_9_train=data_9['x'][:,:].transpose((1,0))
y_9_train=data_9['y'][:,:,:].transpose((2,1,0))
x_10_train=data_10['x'][:,:].transpose((1,0))
y_10_train=data_10['y'][:,:,:].transpose((2,1,0))
x_11_train=data_11['x'][:,:].transpose((1,0))
y_11_train=data_11['y'][:,:,:].transpose((2,1,0))





#____input testing data____#
data2=sio.loadmat(train_folder+'/DSD100_16k_100percentVocal_pairedMix_randomMix_validation.mat')
print('Data loading finish.')
x_test = data2['x'][:,:].transpose((1,0))
y_test = data2['y'][:,:,:].transpose((2,1,0))
x_test = np.expand_dims(x_test, 1)
test_data = Wavedata(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test_data, 10,  shuffle = False)
#Original Mix
x_train = np.concatenate([x_1_train,x_2_train,x_3_train,x_4_train,x_5_train, \
                          x_6_train,x_7_train,x_8_train,x_9_train,x_10_train,x_11_train], axis = 0)
x_train = np.expand_dims(x_train, 1)
y_train = np.concatenate([y_1_train,y_2_train,y_3_train,y_4_train,y_5_train, \
                          y_6_train,y_7_train,y_8_train,y_9_train,y_10_train,y_11_train],axis = 0)
train_data = Wavedata(x_train, y_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size,  shuffle = True)
#%% model save
def save_checkpoint(checkpoint_path, model, optimizer):
    # state_dict: a Python dictionary object that:
    # - for a model, maps each layer to its parameter tensor;
    # - for an optimizer, contains info about the optimizer’s states and hyperparameters used.
    state = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
     
    


model = wenchen_Net.cortex_separator(hidden_size=hidden_size,lstm_hidden_size=lstm_hidden_size,
                                      unidirectional=unidirectional,nb_bins=nb_bins,causal=causal,
                                      use_cortex=use_cortex,num_layers=num_layers).to(device)
print(model)
encoder = wenchen_Net.Encoder_TorchSTFT(n_fft = nfft, n_hop=hop, center=True).to(device)
# encoder_S = wenchen_Net.Encoder_TorchSTFT(n_fft = nfft_S, n_hop=hop, center=True).to(device)
decoder = wenchen_Net.Decoder_TorchISTFT(n_fft = nfft, n_hop=hop, center=True).to(device)
# decoder_S = wenchen_Net.Decoder_TorchISTFT(n_fft = nfft_S, n_hop=hop, center=True).to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

writer = SummaryWriter('./Log_dir')

# ## =========continue training ==============
# Checkpoint = torch.load('./checkpoint/tasnet_89epoch.pth')
# model.load_state_dict(Checkpoint['state_dict'])
# optimizer.load_state_dict(Checkpoint['optimizer'])
# # # ===========================================


# weights = dict()
# for name, param in model.named_parameters():
#     weights[name] = param
#     print(name, param.size(), type(param))






def validation():
    model.eval()  # set evaluation mode
    dataloader_iterator = iter(test_loader)
    loss_spec_append = []
    loss_time_append = []
    loss_append=[]
    SDR_vocal_append=[]
    SDR_music_append=[]
    
    for idx in range(int(len(x_test)/10)):
        x_valid, t_valid = next(dataloader_iterator)
        with torch.no_grad():
            
            X_L = encoder(x_valid.to(device))
            # print('encoder_L:', X_L.shape) #([batch, 1, 513, 63, 2])  #note center = True 
            # X_S = encoder_S(x_valid.to(device))
            # print('encoder_S:', X_S.shape) #([batch, 1, 257, 63, 2])
           
            
            mag_X_L = torch.sqrt(torch.pow(X_L[:,:,:,:,0],2) + torch.pow(X_L[:,:,:,:,1],2)) #[batch,channel,f_bin,t_frame]
            # mag_X_S = torch.sqrt(torch.pow(X_S[:,:,:,:,0],2) + torch.pow(X_S[:,:,:,:,1],2)) #[batch,channel,f_bin,t_frame]
            mag_X =mag_X_L #[batch,channel,f_bin*2,t_frame]
            # print("mag_X.shape: ", mag_X.shape)  #([batch, 1, 770, 63])
    
    
            est_mask,(hn,cn) = model(mag_X) #[batch,vocal/music,f_bin*2,t_frame]
            # print("est_mask.shape: ", est_mask.shape)  #([batch, 2, 770, 63])
            
            
            T_L = encoder(t_valid.to(device))
            # print('T_encoder_L:', T_L.shape) #([batch, 2, 513, 63, 2])
            # T_S = encoder_S(t_valid.to(device))
            # print('T_encoder_S:', T_S.shape) #([batch, 2, 257, 63, 2])
            mag_T_L = torch.sqrt(torch.pow(T_L[:,:,:,:,0],2) + torch.pow(T_L[:,:,:,:,1],2)) #[batch,channel,f_bin,t_frame]
            mag_T = mag_T_L #[batch,channel,f_bin*2,t_frame]
            # print("mag_T.shape: ", mag_T.shape)  #([batch, 2, 770, 63])
 

            loss_vocal_spec = criterion(est_mask[:,0,:,:],mag_T[:,0,:,:].to(device))
            loss_music_spec = criterion(est_mask[:,1,:,:],mag_T[:,1,:,:].to(device))        
            loss_spec = (loss_vocal_spec + loss_music_spec)
            loss_spec_append.append(loss_spec.item())

            X_L_phase = torch.angle(torch.view_as_complex(X_L)).repeat(1,2,1,1)     
            est_mask_L = est_mask
            # print(est_mask_L.shape)
            
            Y_estimate_L = torch.polar(est_mask_L, X_L_phase)
            Y_estimate_L = torch.view_as_real(Y_estimate_L)
            y_estimate_vocal_L = decoder(Y_estimate_L[:,0,:,:,:],16000)
            
            y_estimate_music_L = decoder(Y_estimate_L[:,1,:,:,:],16000)
            # print('shape.mixture_decode_istft_L:'+str(y_estimate_music_L.shape)) #([10, 16000])
            
            y_estimate_vocal = (y_estimate_vocal_L )
            y_estimate_music = (y_estimate_music_L )
            
            loss_vocal_time = criterion(y_estimate_vocal,t_valid[:,0,:].to(device))
            loss_music_time = criterion(y_estimate_music,t_valid[:,1,:].to(device))
            loss_time = loss_vocal_time + loss_music_time
            loss_time_append.append(loss_time.item())
            
            loss = arfa*loss_time + (1-arfa)*loss_spec   
            loss_append.append(loss.item())
            
            
            if idx%3==0:
                vocal_cat = t_valid[:,0,:].numpy()
                music_cat = t_valid[:,1,:].numpy()
                mix_cat = x_valid[:,0,:].numpy()
                estimate_vocal_cat = y_estimate_vocal.cpu().detach().numpy()
                estimate_music_cat = y_estimate_music.cpu().detach().numpy()
                continue
            
            
            estimate_vocal_cat = np.concatenate((estimate_vocal_cat,y_estimate_vocal.cpu().detach().numpy()),0) 
            estimate_music_cat = np.concatenate((estimate_music_cat,y_estimate_music.cpu().detach().numpy()),0) 
            vocal_cat = np.concatenate((vocal_cat,t_valid[:,0,:].numpy()),0)
            music_cat = np.concatenate((music_cat,t_valid[:,1,:].numpy()),0)
            mix_cat = np.concatenate((mix_cat,x_valid[:,0,:].numpy()),0)
            

            
            if (idx+1)%3== 0:
                estimate_vocal_cat = np.reshape(estimate_vocal_cat,[-1])
                estimate_music_cat = np.reshape(estimate_music_cat,[-1])
                vocal_cat = np.reshape(vocal_cat,[-1])
                music_cat = np.reshape(music_cat,[-1])
                mix_cat = np.reshape(mix_cat,[-1])
                
                # wavwrite('debug_signal/vocal_'+str(idx)+'.wav', estimate_vocal_cat, 16000)
                # wavwrite('debug_signal/music_'+str(idx)+'.wav', estimate_music_cat, 16000)
                # wavwrite('debug_signal/cleanmusic_'+str(idx)+'.wav', music_cat, 16000)
                # wavwrite('debug_signal/cleanvocal_'+str(idx)+'.wav', vocal_cat, 16000)
                # wavwrite('debug_signal/mix_'+str(idx)+'.wav', mix_cat, 16000)
                
                SDR_vocal = compute_measures(estimate_vocal_cat,vocal_cat)
                # print('VOCAL SDR',SDR_vocal)
                SDR_music = compute_measures(estimate_music_cat,music_cat)
                # print('Music SDR',SDR_music)
                SDR_vocal_append.append(SDR_vocal)
                SDR_music_append.append(SDR_music)
                
    SDR_vocal_append = np.array(SDR_vocal_append)
    SDR_vocal_append = SDR_vocal_append[np.logical_not(np.isnan(SDR_vocal_append))]
    SDR_music_append = np.array(SDR_music_append)
    SDR_music_append = SDR_music_append[np.logical_not(np.isnan(SDR_music_append))]
    
    # print(np.median(SDR_vocal_append))
    # print(np.median(SDR_music_append))
                
    print ('Epoch [{}/{}],validatio_Loss: {}'.format(epoch+1, epochs_size,np.mean(loss_append)) )       
    print('vocal_SDR', np.median(SDR_vocal_append))
    print('music_SDR', np.median(SDR_music_append))
    model.train()
    return np.mean(loss_append), np.mean(loss_time_append), np.mean(loss_spec_append),np.mean(SDR_vocal_append),np.mean(SDR_music_append)
    


# import MF_model_oneLSTM as wenchen_Net 
# model = wenchen_Net.cortex_separator(nb_bins=nb_bins,causal=True,use_cortex=True).to(device)
# print(model)
# encoder = wenchen_Net.Encoder_TorchSTFT(n_fft = nfft, n_hop=hop, center=True).to(device)
# # encoder_S = wenchen_Net.Encoder_TorchSTFT(n_fft = nfft_S, n_hop=hop, center=True).to(device)
# decoder = wenchen_Net.Decoder_TorchISTFT(n_fft = nfft, n_hop=hop, center=True).to(device)
# # decoder_S = wenchen_Net.Decoder_TorchISTFT(n_fft = nfft_S, n_hop=hop, center=True).to(device)
# [valid_loss, v_vocal_L1loss, v_music_L1loss, SDR_vocal,  SDR_music] = validation()



#%% train
training_loss = []
train_loss_time = []
train_loss_spec = []




validation_loss = []
validation_loss_time = []
validation_loss_spec = []

validation_SDR_vocal = []
validation_SDR_music = []


best_vocal_SDR = 0
best_music_SDR = 0
best_epoch = 0


checkpoint_path= './checkpoint/initial.pth'
save_checkpoint(checkpoint_path,model,optimizer)


total_step = len(train_loader)
print('strat training....')
model.train()
for epoch in range(epochs_size):
    epoch_now = epoch
    start = timeit.default_timer()
    for i, (x,t) in enumerate(train_loader):
        # Forward pass
        # if i==10:
        #     break
        
        X = encoder(x.to(device))
        # print('encoder_L:', X_L.shape) #([batch, 1, 513, 63, 2])  #note center = True 
        mag_X = torch.sqrt(torch.pow(X[:,:,:,:,0],2) + torch.pow(X[:,:,:,:,1],2)) #[batch,channel,f_bin,t_frame]
        est_mask,(hn,cn) = model(mag_X) #[batch,vocal/music,f_bin*2,t_frame]
        # print("est_mask.shape: ", est_mask.shape)  #([batch, 2, 770, 63])
        T = encoder(t.to(device))
        # print('T_encoder_L:', T_L.shape) #([batch, 2, 513, 63, 2])
        mag_T = torch.sqrt(torch.pow(T[:,:,:,:,0],2) + torch.pow(T[:,:,:,:,1],2)) #[batch,channel,f_bin,t_frame]
       
        #===================  creating time loss ===============================
        X_L_phase = torch.angle(torch.view_as_complex(X)).repeat(1,2,1,1)     
        
        est_mask_L = est_mask
        # print(est_mask_S.shape)
        # print(est_mask_L.shape)
        
        Y_estimate_L = est_mask_L*(torch.cos(X_L_phase)+1j*torch.sin(X_L_phase))
        Y_estimate_L = torch.view_as_real(Y_estimate_L)
        # print(Y_estimate_L.size()) #([10, 2, 257, 63, 2])
        y_estimate_vocal_L = decoder(Y_estimate_L[:,0,:,:,:],16000)
        y_estimate_music_L = decoder(Y_estimate_L[:,1,:,:,:],16000)
        # print('shape.mixture_decode_istft_L:'+str(y_estimate_music_L.shape)) #([10, 16000])
        
        y_estimate_vocal = y_estimate_vocal_L 
        y_estimate_music = y_estimate_music_L
        
        loss_vocal_time = criterion(y_estimate_vocal,t[:,0,:].to(device))
        loss_music_time = criterion(y_estimate_music,t[:,1,:].to(device))
        loss_time = loss_vocal_time + loss_music_time
 
        
        

        # print(y_estimate.shape)
        # loss = penalty_L1loss(y_estimate,t.to(device))
        
        loss_vocal_spec = criterion(est_mask[:,0,:,:], mag_T[:,0,:,:].to(device))
        loss_music_spec = criterion(est_mask[:,1,:,:], mag_T[:,1,:,:].to(device))
        
        loss_spec = (loss_vocal_spec + loss_music_spec)
        loss = arfa*loss_time + (1-arfa)*loss_spec
       
        
        
        # test_mag,test_stft = encoder(x.to(device))
        # X_phase = torch.angle(torch.view_as_complex(test_stft)).repeat(1,2,1,1)
        # print(X_phase.size()) #([10, 2, 257, 63])
        # Y_estimate = torch.polar(est_mask, X_phase) 
        # Y_estimate = torch.view_as_real(Y_estimate)
        # # print(Y_estimate.size()) #([10, 2, 257, 63, 2])
            
        # y_estimate_vocal = decoder(Y_estimate[:,0,:,:,:],16000).cpu().detach().numpy() #([10, 15872])
        # y_estimate_music = decoder(Y_estimate[:,1,:,:,:],16000).cpu().detach().numpy() #([10, 15872])
            
        
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {}, ' 
                      .format(epoch_now+1, epochs_size, i+1, total_step, loss.item()))
            print('Best epoch:'+ str(best_epoch)+' Vocal: '+str(best_vocal_SDR)+' Music: '+str(best_music_SDR))
            
            
            
    #== validation =======
    stop = timeit.default_timer()
    print('Time for one epoch :'+ str(stop-start)+' seconds')
    print ('Epoch [{}/{}], Step [{}/{}], Loss: {} ' 
                      .format(epoch_now+1, epochs_size, i+1, total_step, loss.item()))
    
    [valid_loss, valid_loss_time, valid_loss_spec, SDR_vocal, SDR_music] = validation()


    print('is train ? '+str(model.training))
    
    
    validation_loss.append(valid_loss)
    validation_SDR_vocal.append(SDR_vocal)
    validation_SDR_music.append(SDR_music)
    validation_loss_time.append(valid_loss_time*arfa)
    validation_loss_spec.append(valid_loss_spec*(1-arfa))
 
    
    
    
    writer.add_scalar('Validation/Vocal SDR', SDR_vocal, epoch_now)
    writer.add_scalar('Validation/Music SDR', SDR_music, epoch_now)
    writer.add_scalar('Validation Loss/time loss', valid_loss_time, epoch_now)
    writer.add_scalar('Validation Loss/spec Loss', valid_loss_spec, epoch_now)

    
    
    
    training_loss.append(loss.item())
    train_loss_time.append(loss_time.item()*arfa)
    train_loss_spec.append(loss_spec.item()*(1-arfa))

    writer.add_scalar('Loss/Training_Loss', loss, epoch_now)
    writer.add_scalar('Loss/validation_Loss', valid_loss, epoch_now)
    writer.add_scalar('Loss/vocal_spec_L1Loss', loss_vocal_spec, epoch_now)
    writer.add_scalar('Loss/music_spec_L1Loss', loss_music_spec, epoch_now)
    writer.add_scalar('Loss/vocal_time_L1Loss', loss_vocal_time, epoch_now)
    writer.add_scalar('Loss/music_time_L1Loss', loss_music_time, epoch_now)
    # writer.add_scalar('Loss/vocal_Dissimilarity', dis_vocal, epoch_now)
    # writer.add_scalar('Loss/music_Dissimilarity', dis_music, epoch_now)

    
    plt.plot(validation_SDR_vocal,label = "Vocal SDR")
    plt.plot(validation_SDR_music,label = "Music SDR")
    plt.legend()
    plt.show()
    
    plt.plot(training_loss,label = "training loss")
    plt.legend()
    plt.show()
    
    plt.plot(train_loss_time,label = "training loss time X arfa")
    plt.plot(train_loss_spec,label = "training loss spec X 1-arfa")
    plt.legend()
    plt.show()
    
    
    
    plt.plot(validation_loss,label = "validation Loss")
    plt.legend()
    plt.show()
    
    plt.plot(validation_loss_time,label = "validation Loss time X arfa")
    plt.plot(validation_loss_spec,label = "validation Loss spec X (1-arfa)")
    plt.legend()
    plt.show()
    

    
    checkpoint_path= './checkpoint/tasnet_'+str(epoch_now)+'epoch.pth'
    save_checkpoint(checkpoint_path,model,optimizer)

    
    if (SDR_vocal>best_vocal_SDR) & (SDR_music>best_music_SDR):
        best_vocal_SDR = SDR_vocal
        best_music_SDR = SDR_music
        best_epoch = epoch_now
        checkpoint_path='./checkpoint/tasnet_bestSDR_'+str(best_epoch)+'epoch.pth'
        save_checkpoint(checkpoint_path,model,optimizer)


    checkpoint_path= './checkpoint/tasnet_'+str(epoch_now)+'epoch.pth'
    save_checkpoint(checkpoint_path,model,optimizer)
sio.savemat('training_loss.mat', {'loss': np.array(training_loss)})