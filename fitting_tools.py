import torch
from torch import nn
import numpy as np
import os
from secrets import token_urlsafe

class Dataloader_iterations:
    def __init__(self, data, batch_size, iterations):
        self.data = [torch.as_tensor(d,dtype=torch.float32) for d in data] #this copies the data again
        self.batch_size = batch_size
        self.iterations = iterations
    
    def __iter__(self):
        return Dataloader_iterationsIterator(self.data, self.batch_size, self.iterations)
    def __len__(self):
        return self.iterations
    
class Dataloader_iterationsIterator:
    def __init__(self, data, batch_size, iterations):
        self.ids = np.arange(len(data[0]),dtype=int)
        self.data = data
        self.L = len(data[0])
        self.batch_size = self.L if batch_size>self.L else batch_size            
        self.data_counter = 0
        self.it_counter = 0
        self.iterations = iterations

    def __iter__(self):
        return self
    def __next__(self):
        self.it_counter += 1
        if self.it_counter>self.iterations:
            raise StopIteration
        self.data_counter += self.batch_size
        ids_now = self.ids[self.data_counter-self.batch_size:self.data_counter]
        if self.data_counter+self.batch_size>self.L: #going over the limit next time, hence, shuffle and restart
            self.data_counter = 0
            np.random.shuffle(self.ids)
        return [d[ids_now] for d in self.data]

from copy import deepcopy
from tqdm.auto import tqdm
class Fitting_nnModule(nn.Module):
    def fit(self, train, val, iterations=10_000, batch_size=256, loss_kwargs={}, \
            print_freq=100, loss_kwargs_val=None, call_back_validation=None, val_freq=None, optimizer=None,\
            save_freq=None, save_file=None):
        '''The main fitting function        
        '''
        loss_kwargs_val = (loss_kwargs if loss_kwargs_val is None else loss_kwargs_val)
        if call_back_validation is None:
            val_data = self.make_training_data(val, **loss_kwargs_val)
        train_data = self.make_training_data(train, **loss_kwargs)
        print('Number of datapoints:', len(train_data[0]), '\tBatch size: ', batch_size, '\tIterations per epoch:', len(train_data[0])//batch_size)
        optimizer = torch.optim.Adam(self.parameters()) if optimizer is None else optimizer #optimizer is not a part of the Module

        #monitoring and checkpoints
        if not hasattr(self, 'loss_train_monitor'):
            self.loss_train_monitor, self.loss_val_monitor, self.iteration_monitor = [], [], []
            iteration_counter_offset = 0
        else:
            print('Restarting training!!!, this might result in weird behaviour')
            iteration_counter_offset = self.iteration_monitor[-1]
        lowest_train_loss, loss_train_acc, _ = float('inf'), 0, self.checkpoint_save('lowest_train_loss')
        lowest_val_loss, loss_val, _ = float('inf'), float('inf'), self.checkpoint_save('lowest_val_loss')
        val_freq  = print_freq if val_freq==None  else val_freq
        save_freq = print_freq if save_freq==None else save_freq
        if save_file is None and save_freq!=False:
            code = token_urlsafe(4).replace('_','0').replace('-','a')
            save_file = os.path.join(get_checkpoint_dir(), f'{self.__class__.__name__}-{code}.pth')
    
        data_iter = enumerate(tqdm(Dataloader_iterations(train_data, batch_size=batch_size, iterations=iterations), initial=1),start=1)
        try:
            for iteration, batch in data_iter:
                loss = self.loss(*batch,**loss_kwargs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_train_acc += loss.item()
                
                # Saving and printing
                if iteration%val_freq==0:
                    loss_val = self.loss(*val_data, **loss_kwargs_val).item() if \
                        call_back_validation is None else call_back_validation(locals(), globals())
                    if loss_val<lowest_val_loss:
                        lowest_val_loss = loss_val
                        self.checkpoint_save('lowest_val_loss')
                if iteration%print_freq==0:
                    loss_train = loss_train_acc/print_freq
                    m = '!' if loss_train<lowest_train_loss else ' '
                    M = '!' if len(self.loss_val_monitor)==0 or np.min(self.loss_val_monitor)>lowest_val_loss else ' '
                    print(f'it {iteration:7,} loss {loss_train:.3f}{m} loss val {loss_val:.3f}{M}')
                    self.loss_train_monitor.append(loss_train)
                    self.loss_val_monitor.append(loss_val)
                    self.iteration_monitor.append(iteration+iteration_counter_offset)
                    if loss_train<lowest_train_loss:
                        lowest_train_loss = loss_train
                        self.checkpoint_save('lowest_train_loss')
                    loss_train_acc = 0
                if save_freq!=False and (iteration%save_freq==0):
                    self.save_to_file(save_file)

                
        except KeyboardInterrupt:
            print('stopping early, ', end='')
        print('Saving parameters to checkpoint self.checkpoints["last"] and loading self.checkpoints["lowest_val_loss"]')
        self.checkpoint_save('last')
        self.checkpoint_load('lowest_val_loss') #Should this also save the monitors?
        if save_freq!=False:
            self.save_to_file(save_file)
    
    def checkpoint_save(self,name): #checkpoints do not use files
        if not hasattr(self, 'checkpoints'):
            self.checkpoints = {}
        self.checkpoints[name] = deepcopy(self.state_dict())

    def checkpoint_load(self, name):
        self.load_state_dict(self.checkpoints[name])
    
    def save_to_file(self, file):
        torch.save(self, file)

class constant_net(nn.Module):
    def __init__(self, n_out=5, bias_scale=1.):
        super().__init__()
        self.n_out = n_out
        self.bias = nn.Parameter(bias_scale*(torch.rand(n_out)*2-1)*3**0.5) #init such that it is uniform with a std of 1
    
    def forward(self, x):
        return torch.broadcast_to(self.bias, (x.shape[0], self.n_out))

class MLP_res_net(nn.Module): #a simple MLP
    def __init__(self,n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh, zero_bias=True, bias_scale=1.):
        super(MLP_res_net, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        if n_hidden_layers>0 and self.n_in!=0:
            seq = [nn.Linear(n_in,n_nodes_per_layer),activation()]
            for i in range(n_hidden_layers-1):
                seq.append(nn.Linear(n_nodes_per_layer,n_nodes_per_layer))
                seq.append(activation())
            seq.append(nn.Linear(n_nodes_per_layer,n_out))
            self.net = nn.Sequential(*seq)
        else:
            self.net = None
        
        self.net_lin = nn.Linear(n_in, n_out) if n_in>0 else constant_net(n_out,bias_scale=bias_scale)
        if zero_bias:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.bias, val=0) #bias
        
    def forward(self,X):
        if self.net is None:
            return self.net_lin(X)
        else:
            return self.net(X) + self.net_lin(X)

class Norm:
    def __init__(self, u=None, y=None, umean=0., ustd=1., ymean=0., ystd=1.):
        if u is None:
            self.umean = umean
            self.ustd = ustd
        else:
            self.input_norm_fit(u) #self.umean = #this as tensors or as numpy arrays?
        if y is None:    
            self.ymean = ymean
            self.ystd = ystd
        else:
            self.output_norm_fit(y)
    def input_norm_fit(self, u):
        self.umean = np.mean(u,axis=0)
        self.ustd = np.std(u,axis=0)
    def output_norm_fit(self, y):
        self.ymean = np.mean(y,axis=0)
        self.ystd = np.std(y,axis=0)
    def fit(self, u, y):
        self.input_norm_fit(u)
        self.output_norm_fit(y)
    def output_transform(self, y): #un-normlized -> normalized
        return (y-self.ymean)/self.ystd
    def input_transform(self, u): #un-normlized -> normalized
        return (u-self.umean)/self.ustd
    def output_inverse_transform(self, y): #un-normlized -> normalized
        return y*self.ystd + self.ymean #(y-self.ymean)/self.ystd
    def input_inverse_transform(self, u): #un-normlized -> normalized
        return u*self.ustd + self.umean #(u-self.umean)/self.ustd

def get_nuy_and_auto_norm(u, y):
    nu = None if u.ndim==1 else u.shape[1]
    ny = None if y.ndim==1 else y.shape[1]
    norm = Norm(u, y)
    return nu, ny, norm

def get_checkpoint_dir():
    '''A utility function which gets the checkpoint directory for each OS

    It creates a working directory called meta-SS-checkpoints 
        in LOCALAPPDATA/meta-SS-checkpoints/ for windows
        in ~/.meta-SS-checkpoints/ for unix like
        in ~/Library/Application Support/meta-SS-checkpoints/ for darwin

    Returns
    -------
    checkpoints_dir
    '''
    def mkdir(directory):
        if os.path.isdir(directory) is False:
            os.mkdir(directory)
    from sys import platform
    if platform == "darwin": #not tested but here it goes
        checkpoints_dir = os.path.expanduser('~/Library/Application Support/meta-SS-checkpoints/')
    elif platform == "win32":
        checkpoints_dir = os.path.join(os.getenv('LOCALAPPDATA'),'meta-SS-checkpoints/')
    else: #unix like, might be problematic for some weird operating systems.
        checkpoints_dir = os.path.expanduser('~/.meta-SS-checkpoints/')#Path('~/.deepSI/')
    mkdir(checkpoints_dir)
    return checkpoints_dir

if __name__=='__main__' and False:
    data = [np.random.rand(40,2), np.random.rand(40)]
    batch_size = 11
    iterator = Dataloader_iterations(data, batch_size, iterations=10)
    for it,d in enumerate(iterator,start=1):
        print(it, d[0].shape, d[1].shape)

if __name__=='__main__' and True:
    net = MLP_res_net(0, 5)
    print(net(torch.rand(10,0)))

