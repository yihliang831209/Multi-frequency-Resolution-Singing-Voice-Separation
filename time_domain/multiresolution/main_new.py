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
from tasnet_v2_1 import ConvTasNet
from utils import sizeof_fmt
import numpy as np
import timeit
from torch.utils.tensorboard import SummaryWriter 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

#____Input training data____#
print('Data loading....')
train_folder = '../../../dataset/'
data_1 = sio.loadmat(train_folder+'DSD100_16k_100percentVocal_pairedMix_randomMix_1.mat')
data_2 = sio.loadmat(train_folder+'DSD100_16k_100percentVocal_pairedMix_randomMix_2.mat')
data_3 = sio.loadmat(train_folder+'DSD100_16k_100percentVocal_pairedMix_randomMix_3.mat')
data_4 = sio.loadmat(train_folder+'DSD100_16k_100percentVocal_pairedMix_randomMix_4.mat')
data_5 = sio.loadmat(train_folder+'DSD100_16k_100percentVocal_pairedMix_randomMix_5.mat')
data_6 = sio.loadmat(train_folder+'DSD100_16k_100percentVocal_pairedMix_randomMix_6.mat')
data_7 = sio.loadmat(train_folder+'DSD100_16k_100percentVocal_pairedMix_randomMix_7.mat')
data_8 = sio.loadmat(train_folder+'DSD100_16k_100percentVocal_pairedMix_randomMix_8.mat')
data_9 = sio.loadmat(train_folder+'DSD100_16k_100percentVocal_pairedMix_randomMix_9.mat')
data_10 = sio.loadmat(train_folder+'DSD100_16k_100percentVocal_pairedMix_randomMix_10.mat')
data_11 = sio.loadmat(train_folder+'DSD100_16k_100percentVocal_pairedMix_randomMix_11.mat')


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
data2=sio.loadmat(train_folder+'DSD100_16k_100percentVocal_pairedMix_randomMix_validation.mat')
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



#%%

model = ConvTasNet()
print(model)
size = sizeof_fmt(4 * sum(p.numel() for p in model.parameters()))
print(f"Model size {size}")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = nn.L1Loss()

### =========continue training ==============
Checkpoint = torch.load(r'./checkpoint/tasnet_55epoch.pth')
model.load_state_dict(Checkpoint['state_dict'])
optimizer.load_state_dict(Checkpoint['optimizer'])
### ===========================================

# def test():
#     model.eval()  # set evaluation mode
#     test_loss = 0
#     with torch.no_grad():
#         for x, t in test_loader:
#             x_power =x.to(device)
#             t_power = t.to(device)
#             #x_power = (np.conj(x)*x).real.to(device)
#             #t_power = (np.conj(t)*t).real.to(device)
#             # Forward pass
#             x_mask = model(x_power).to(device)
#             output = x_power[:,65:130]*x_mask
#             test_loss += criterion(output, t_power).item()
#     test_loss/=len(test_loader.dataset)
#     testing_loss.append(test_loss)    
def validation():
    model.eval()  # set evaluation mode
    dataloader_iterator = iter(test_loader)
    loss_append = []
    SDR_vocal_append=[]
    SDR_music_append=[]
    for idx in range(int(len(x_test)/10)):
        x_valid, t_valid = next(dataloader_iterator)
        with torch.no_grad():
            y_estimate = model(x_valid.to(device))      
            loss = criterion(y_estimate,t_valid.to(device))
            loss_append.append(loss.item())
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
                
    print ('Epoch [{}/{}],validatio_Loss: {}'.format(epoch+1, epochs_size,np.mean(loss_append)) )        
    model.train()
    return np.mean(loss_append),np.mean(SDR_vocal_append),np.mean(SDR_music_append)
#%% train
training_loss = []
testing_loss = []
validation_loss = []
validation_SDR_vocal = []
validation_SDR_music = []
best_vocal_SDR = 0
best_music_SDR = 0
best_mean_SDR = 0
best_epoch = 0
### initial summary writter #####################
writer = SummaryWriter('log_dir')



total_step = len(train_loader)
print('strat training....')
model.train()
for epoch in range(epochs_size):
    epoch_now = epoch+56
    start = timeit.default_timer()
    train_loss_sum = []
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
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {}' 
                      .format(epoch_now+1, epochs_size, i+1, total_step, loss.item()))
            print('Best epoch:'+ str(best_epoch)+' Vocal: '+str(best_vocal_SDR)+' Music: '+str(best_music_SDR) )
    #== validation =======
    stop = timeit.default_timer()
    print('Time for one epoch :'+ str(stop-start)+' seconds')
    print ('Epoch [{}/{}], Step [{}/{}], Loss: {}' 
                      .format(epoch_now+1, epochs_size, i+1, total_step, loss.item()))
    
    [valid_loss,SDR_vocal_DSD100,SDR_music_DSD100] = validation()
    print('is train ? '+str(model.training))
    validation_loss.append(valid_loss)
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
    
    plt.plot(validation_loss,label = "validation Loss")
    plt.legend()
    plt.show()
    writer.add_scalar('Validation/loss',valid_loss,epoch_now)
    
    plt.plot(training_loss,label = "training loss")
    plt.legend()
    plt.show()
    writer.add_scalar('Train/loss',np.mean(train_loss_sum),epoch_now)
    
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