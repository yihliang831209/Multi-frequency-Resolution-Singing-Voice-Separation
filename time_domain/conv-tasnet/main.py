# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 23:10:08 2020

@author: User
"""
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import scipy.io as sio
import torch 
from torch import nn
from tasnet import ConvTasNet
from utils import sizeof_fmt
import numpy as np
import timeit
from torch.utils.tensorboard import SummaryWriter 
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

batch_size = 10
epochs_size = 100
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
#____input testing data____#
train_folder = '../../../dataset/'
data2=sio.loadmat(train_folder+'DSD100_16k_100percentVocal_pairedMix_randomMix_validation.mat')
print('Data loading finish.')
x_test = data2['x'][:,:].transpose((1,0))
y_test = data2['y'][:,:,:].transpose((2,1,0))
x_test = np.expand_dims(x_test, 1)
len_x_test =len(x_test)
test_data = Wavedata(x_test, y_test)
test_loader = torch.utils.data.DataLoader(test_data, 10,  shuffle = False)
#%% release memory
del x_test
del y_test
del data2
#Original Mix

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



#%%

model = ConvTasNet()
print(model)
size = sizeof_fmt(4 * sum(p.numel() for p in model.parameters()))
print(f"Model size {size}")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.L1Loss()

### =========continue training ==============
# Checkpoint = torch.load(r'./checkpoint/tasnet_55epoch.pth')
# model.load_state_dict(Checkpoint['state_dict'])
# optimizer.load_state_dict(Checkpoint['optimizer'])
### ===========================================
def validation():
    model.eval()  # set evaluation mode
    dataloader_iterator = iter(test_loader)
    L1loss_append = []
    SDR_vocal_append=[]
    SDR_music_append=[]
    for idx in range(len_x_test//10):
        x_valid, t_valid = next(dataloader_iterator)
        with torch.no_grad():
            y_estimate = model(x_valid.to(device))      
            loss = criterion(y_estimate,t_valid.to(device))
            L1loss_append.append(loss.item())
            if idx%3==0:
                vocal_cat = t_valid[:,0,:].numpy()
                music_cat = t_valid[:,1,:].numpy()
                estimate_vocal_cat = y_estimate[:,0,:].cpu().detach().numpy()
                estimate_music_cat = y_estimate[:,1,:].cpu().detach().numpy()
                continue
            
            estimate_vocal_cat = np.concatenate((estimate_vocal_cat,y_estimate[:,0,:].cpu().detach().numpy()),0) 
            estimate_music_cat = np.concatenate((estimate_music_cat,y_estimate[:,1,:].cpu().detach().numpy()),0) 
            vocal_cat = np.concatenate((vocal_cat,t_valid[:,0,:].numpy()),0)
            music_cat = np.concatenate((music_cat,t_valid[:,1,:].numpy()),0)
            
          
            if (idx+1)%3== 0:
                estimate_vocal_cat = np.reshape(estimate_vocal_cat,[-1])
                estimate_music_cat = np.reshape(estimate_music_cat,[-1])
                vocal_cat = np.reshape(vocal_cat,[-1])
                music_cat = np.reshape(music_cat,[-1])
                SDR_vocal = compute_measures(estimate_vocal_cat,vocal_cat)
                SDR_music = compute_measures(estimate_music_cat,music_cat)
                SDR_vocal_append.append(SDR_vocal)
                SDR_music_append.append(SDR_music)
                
    print ('Epoch [{}/{}],validatio_Loss: {}'.format(epoch+1, epochs_size,np.mean(L1loss_append)) )        
    model.train()
    return np.mean(L1loss_append),np.median(SDR_vocal_append),np.median(SDR_music_append)
#%% train
training_loss = []
testing_loss = []
validation_L1loss = []
validation_SDR_vocal = []
validation_SDR_music = []
best_vocal_SDR = 0
best_music_SDR = 0
best_mean_SDR = 0
best_epoch = 0
### initial summary writter #####################
writer = SummaryWriter('log_dir')
print('strat training....')
model.train()
for epoch in range(epochs_size):
    start = timeit.default_timer()
    epoch_now = epoch
    file_sequence = np.random.permutation(11)
    train_loss_sum = []
    for load_file_i in range(3):  ## separate all data into several parts
        print('Data loading '+str(load_file_i+1)+'/3 ....')
        data_1 = sio.loadmat(train_folder+'DSD100_16k_100percentVocal_pairedMix_randomMix_'+str(file_sequence[3*load_file_i]+1)+'.mat')
        data_2 = sio.loadmat(train_folder+'DSD100_16k_100percentVocal_pairedMix_randomMix_'+str(file_sequence[3*load_file_i+1]+1)+'.mat')
        data_3 = sio.loadmat(train_folder+'DSD100_16k_100percentVocal_pairedMix_randomMix_'+str(file_sequence[3*load_file_i+2]+1)+'.mat')
        x_1_train=data_1['x'][:,:].transpose((1,0))
        y_1_train=data_1['y'][:,:,:].transpose((2,1,0))
        x_2_train=data_2['x'][:,:].transpose((1,0))
        y_2_train=data_2['y'][:,:,:].transpose((2,1,0))
        x_3_train=data_3['x'][:,:].transpose((1,0))
        y_3_train=data_3['y'][:,:,:].transpose((2,1,0))
        x_train = np.concatenate([x_1_train,x_2_train,x_3_train], axis = 0)
        y_train = np.concatenate([y_1_train,y_2_train,y_3_train],axis = 0)
        del data_1,data_2,data_3
        del x_1_train,x_2_train,x_3_train
        del y_1_train,y_2_train,y_3_train
        x_train = np.expand_dims(x_train, 1)
        train_data = Wavedata(x_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size,  shuffle = True)
        total_step = len(train_loader)   
        for i, (x,t) in enumerate(train_loader):
            # Forward pass
            y_estimate = model(x.to(device))      
            loss = criterion(y_estimate,t.to(device))
            train_loss=loss.item()
            train_loss_sum.append(train_loss)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Data_segament [{}/{}] ,Step [{}/{}], Loss: {}' 
                          .format(epoch_now+1, epochs_size, load_file_i+1, 3, i+1, total_step, loss.item()))
                print('Best epoch:'+ str(best_epoch)+' Vocal: '+str(best_vocal_SDR)+' Music: '+str(best_music_SDR) )
    #== validation =======
    stop = timeit.default_timer()
    print('Time for one epoch :'+ str(stop-start)+' seconds')
    print ('Epoch [{}/{}], Step [{}/{}], Loss: {}' 
                      .format(epoch_now+1, epochs_size, i+1, total_step, loss.item()))
    
    [valid_L1Loss,SDR_vocal_DSD100,SDR_music_DSD100] = validation()
    print('is train ? '+str(model.training))
    validation_L1loss.append(valid_L1Loss)
    validation_SDR_vocal.append(SDR_vocal_DSD100)
    validation_SDR_music.append(SDR_music_DSD100)

    
    training_loss.append(np.mean(train_loss_sum))
    
    plt.plot(validation_SDR_vocal,label = "Vocal SDR")
    plt.plot(validation_SDR_music,label = "Music SDR")
    plt.legend()
    plt.show()
    writer.add_scalar('Validation/DSD100_Vocal_SDR',SDR_vocal_DSD100,epoch_now)
    writer.add_scalar('Validation/DSD100_Music_SDR',SDR_music_DSD100,epoch_now)
    # writer.add_scalar('Validation/ikala_Vocal_SDR',SDR_vocal_ikala,epoch_now)
    # writer.add_scalar('Validation/ikala_Music_SDR',SDR_music_ikala,epoch_now)
    
    plt.plot(validation_L1loss,label = "validation L1 Loss")
    plt.legend()
    plt.show()
    writer.add_scalar('Validation/loss',valid_L1Loss,epoch_now)
    
    plt.plot(training_loss,label = "training L1 loss")
    plt.legend()
    plt.show()
    writer.add_scalar('Train/L1loss',np.mean(train_loss_sum),epoch_now)
    
    if (SDR_vocal_DSD100>best_vocal_SDR) & (SDR_music_DSD100>best_music_SDR):
        best_vocal_SDR = SDR_vocal_DSD100
        best_music_SDR = SDR_music_DSD100
        best_epoch = epoch_now
        checkpoint_path='./checkpoint/tasnet_bestSDR_'+str(best_epoch)+'epoch.pth'
        save_checkpoint(checkpoint_path,model,optimizer)
    if  (SDR_vocal_DSD100+SDR_music_DSD100)/2>best_mean_SDR:
        best_mean_SDR = (SDR_vocal_DSD100+SDR_music_DSD100)/2
        best_epoch = epoch_now
        checkpoint_path='./checkpoint/tasnet_bestMeanSDR_'+str(best_epoch)+'epoch.pth'
        save_checkpoint(checkpoint_path,model,optimizer)
    
    checkpoint_path='./checkpoint/tasnet_'+str(epoch_now)+'epoch.pth'
    save_checkpoint(checkpoint_path,model,optimizer)
checkpoint_path='./checkpoint/tasnet_'+str(epoch_now)+'epoch.pth'
save_checkpoint(checkpoint_path,model,optimizer)