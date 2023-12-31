
# my creations
from fancy_distributions import Multimodal_MultivariateNormal, Multimodal_Normal, stack
from fitting_tools import Fitting_nnModule, MLP_res_net, get_nuy_and_auto_norm, Norm
from static_distribution_modeling import Dependent_dist
import random

from time import time
import numpy as np
import torch
from torch import nn

#todo: 
# nf='sim'
# giving measures for the multi-step-ahead for the validation (given to Multi_step_result?)
# Inherets from Meta_SS_model creates an additional 
#  This becomes a multiple inherentece problem?

class Meta_SS_model(Fitting_nnModule):
    def __init__(self, nu, ny, norm: Norm = Norm(), nz=5, n_weights=10, \
        meta_state_advance_net = MLP_res_net, meta_state_advance_kwargs = {},
        meta_state_to_output_dist_net = Dependent_dist, meta_state_to_output_dist_kwargs=dict(n_weights=10)):
        super(Meta_SS_model, self).__init__()
        self.nz = nz
        self.ny = ny
        self.nu = nu
        self.ny_val = 1 if self.ny is None else self.ny
        self.nu_val = 1 if self.nu is None else self.nu
        self.norm = norm
        self.meta_state_advance = meta_state_advance_net(n_in=nz + self.nu_val, n_out=nz, \
                                    **meta_state_advance_kwargs)
        self.meta_state_to_output_dist = meta_state_to_output_dist_net(nz=nz, ny=ny, norm=Norm(), \
                                  **meta_state_to_output_dist_kwargs) #no norm here?

    def make_training_data(self, uy, nf=50, parameter_init=True, **kwargs):
        u,y = uy
        u = self.norm.input_transform(u)
        y = self.norm.output_transform(y)
        U, Y = [], []
        for i in range(nf,len(u)+1):
            U.append(u[i-nf:i])
            Y.append(y[i-nf:i])
        self.init_z = nn.Parameter(torch.randn((len(U), self.nz))) if parameter_init else torch.zeros((len(U), self.nz))
        ufuture = torch.as_tensor(np.array(U), dtype=torch.float32)
        yfuture = torch.as_tensor(np.array(Y), dtype=torch.float32)
        return ufuture, yfuture, self.init_z
    
    def simulate(self, ufuture, yfuture, init_z, return_z=False):
        zt = init_z
        ydist_preds = []
        ufuture = ufuture[:,:,None] if self.nu is None else ufuture
        zvecs = [] #(Ntime, Nb, nz)
        for ut, yt in zip(torch.transpose(ufuture,0,1),torch.transpose(yfuture,0,1)):
            ydist_pred = self.meta_state_to_output_dist.get_dist_normed(zt)
            ydist_preds.append(ydist_pred)
            zvecs.append(zt) 
            zu = torch.cat([zt, ut], dim=1)
            zt = self.meta_state_advance(zu)
        ydist_preds = stack(ydist_preds,dim=1) #size is (Nbatch, )
        return (ydist_preds, torch.stack(zvecs,dim=1)) if return_z else ydist_preds

    def loss(self, u, y, init_z, burn_time=0, **kwargs):
        ydist_preds = self.simulate(u, y, init_z)
        return torch.mean(-ydist_preds[:,burn_time:].log_prob(y[:,burn_time:]))/self.ny_val + - 1.4189385332046727417803297364056176

    def multi_step(self, uydata, nf=100):
        nf = len(uydata[0]) if nf=='sim' else nf
        ufuture, yfuture, init_z = self.make_training_data(uydata, nf=nf)
        with torch.no_grad():
            y_dists, zfuture = self.simulate(ufuture, yfuture, init_z, return_z=True)
        I = self.norm.input_inverse_transform
        O = self.norm.output_inverse_transform
        test_norm = Norm(uydata[0], uydata[1])
        return Multi_step_result(O(yfuture), O(y_dists), test_norm, data=uydata, ufuture=I(ufuture), zfuture=zfuture)

class Meta_SS_model_encoder(Meta_SS_model):
    def __init__(self, nu: int, ny: int, norm: Norm = Norm(), nz: int=5, na: int=6, nb: int=6,\
                meta_state_advance_net = MLP_res_net, meta_state_advance_kwargs = {},\
                meta_state_to_output_dist_net = Dependent_dist, meta_state_to_output_dist_kwargs=dict(n_weights=10),\
                past_to_meta_state_net = MLP_res_net, past_to_meta_state_kwargs={}):
        super(Meta_SS_model_encoder, self).__init__(nu, ny, norm, nz, meta_state_advance_net=meta_state_advance_net,\
                meta_state_advance_kwargs=meta_state_advance_kwargs, meta_state_to_output_dist_net=meta_state_to_output_dist_net,\
                    meta_state_to_output_dist_kwargs=meta_state_to_output_dist_kwargs)
        self.na = na
        self.nb = nb
        self.past_to_meta_state = past_to_meta_state_net(self.nu_val*nb+self.ny_val*na, nz, **past_to_meta_state_kwargs)
    def make_training_data(self, uy, nf=50, **kwargs):
        u,y = uy
        u = self.norm.input_transform(u)
        y = self.norm.output_transform(y)
        ufuture, yfuture, upast, ypast = [], [], [], []
        for i in range(nf+max(self.na, self.nb),len(u)+1):
            ufuture.append(u[i-nf:i])
            yfuture.append(y[i-nf:i])
            upast.append(u[i-nf-self.nb:i-nf])
            ypast.append(y[i-nf-self.na:i-nf]) #here was the error u -> y
        as_tensor = lambda x: [torch.as_tensor(np.array(xi), dtype=torch.float32) for xi in x]
        return as_tensor([upast, ypast, ufuture, yfuture])
    def simulate(self, upast, ypast, ufuture, yfuture, return_z=False):
        Nb = upast.shape[0]
        init_z = self.past_to_meta_state(torch.cat([upast.view(Nb,-1), ypast.view(Nb,-1)],dim=1))
        return super().simulate(ufuture, yfuture, init_z, return_z=return_z)
    def loss(self, upast, ypast, ufuture, yfuture, **kwargs):
        ydist_preds = self.simulate(upast, ypast, ufuture, yfuture)
        return torch.mean(-ydist_preds.log_prob(yfuture))/self.ny_val + - 1.4189385332046727417803297364056176
    def multi_step(self, uydata, nf=100):
        nf = len(uydata[0])-max(self.na, self.nb) if nf=='sim' else nf
        upast, ypast, ufuture, yfuture = self.make_training_data(uydata, nf=nf)
        with torch.no_grad():
            y_dists, zfuture = self.simulate(upast, ypast, ufuture, yfuture, return_z=True)
        I = self.norm.input_inverse_transform
        O = self.norm.output_inverse_transform
        test_norm = Norm(uydata[0], uydata[1])
        return Multi_step_result(O(yfuture), O(y_dists), test_norm, \
            data=uydata, ufuture=I(ufuture), upast=I(upast), ypast=O(ypast), zfuture=zfuture)

class Meta_SS_model_measure_encoder(Meta_SS_model_encoder): #always includes an encoder
    '''
    z_t+1^t = f^t(z_t^m, u_t)
    z_t+1^m = f^m(z_t+1^t, y_t+1)
    p(y_t| z_t^t) is the output (if z_t^m than it model could be a identity and be oke)
    '''
    def __init__(self, nu: int, ny: int, norm: Norm = Norm(), nz: int=5, na: int=6, nb: int=6,\
                meta_state_advance_net = MLP_res_net, meta_state_advance_kwargs = {},\
                meta_state_to_output_dist_net = Dependent_dist, meta_state_to_output_dist_kwargs=dict(n_weights=10),\
                past_to_meta_state_net = MLP_res_net, past_to_meta_state_kwargs={}, 
                measure_update_net = MLP_res_net, measure_update_kwargs={}):
        super(Meta_SS_model_measure_encoder, self).__init__(nu, ny, norm, nz, na, nb, meta_state_advance_net=meta_state_advance_net,\
                meta_state_advance_kwargs=meta_state_advance_kwargs, meta_state_to_output_dist_net=meta_state_to_output_dist_net,\
                meta_state_to_output_dist_kwargs=meta_state_to_output_dist_kwargs, past_to_meta_state_net=past_to_meta_state_net,\
                past_to_meta_state_kwargs=past_to_meta_state_kwargs)
        ny_val = 1 if ny is None else ny
        self.measure_update = measure_update_net(n_in=ny_val+nz, n_out=nz, **measure_update_kwargs)
    def loss(self, upast, ypast, ufuture, yfuture, filter_p = 1, **kwargs):
        ydist_preds = self.filter(upast, ypast, ufuture, yfuture, filter_p=filter_p)
        return torch.mean(-ydist_preds.log_prob(yfuture))/self.ny_val + - 1.4189385332046727417803297364056176
    def filter(self, upast, ypast, ufuture, yfuture, filter_p = 1, return_z=False, sample_filter_p = 0): #a bit of duplicate code but it's fine. 
        Nb = upast.shape[0]
        zt = self.past_to_meta_state(torch.cat([upast.view(Nb,-1), ypast.view(Nb,-1)],dim=1))
        ydist_preds = []
        ufuture = ufuture[:,:,None] if self.nu is None else ufuture
        yfuture = yfuture[:,:,None] if self.ny is None else yfuture
        zvecs = [] #(Ntime, Nb, nz)
        for ut, yt in zip(torch.transpose(ufuture,0,1),torch.transpose(yfuture,0,1)):
            ydist_pred = self.meta_state_to_output_dist.get_dist_normed(zt)
            ydist_preds.append(ydist_pred)
            zvecs.append(zt)
            if random.random()<filter_p: #filter using output
                if random.random()<sample_filter_p:
                    ysamp = ydist_pred.sample()
                else:
                    ysamp = yt
                zy = torch.cat([zt, ysamp], dim=1)
                zm = self.measure_update(zy)
            else: #skip filter step and recover old implementation
                zm = zt
            zu = torch.cat([zm, ut], dim=1)
            zt = self.meta_state_advance(zu)
        ydist_preds = stack(ydist_preds,dim=1) #size is (Nbatch, )
        return (ydist_preds, torch.stack(zvecs,dim=1)) if return_z else ydist_preds
    def multi_step(self, uydata, nf=100, filter_p = 1, sample_filter_p = 0):
        nf = len(uydata[0])-max(self.na, self.nb) if nf=='sim' else nf
        upast, ypast, ufuture, yfuture = self.make_training_data(uydata, nf=nf)
        with torch.no_grad():
            y_dists, zfuture = self.filter(upast, ypast, ufuture, yfuture, return_z=True, \
                filter_p=filter_p, sample_filter_p=sample_filter_p)
        I = self.norm.input_inverse_transform
        O = self.norm.output_inverse_transform
        test_norm = Norm(uydata[0], uydata[1])
        return Multi_step_result(O(yfuture), O(y_dists), test_norm, \
            data=uydata, ufuture=I(ufuture), upast=I(upast), ypast=O(ypast), zfuture=zfuture)

class Multi_step_result():
    def __init__(self, yfuture, y_dists, norm : Norm, **kwargs):
        #shape = [Nb, Nt, ny] and such
        self.kwargs = kwargs
        self.yfuture = yfuture
        self.y_dists = y_dists
        assert isinstance(norm, Norm)
        self.norm = norm #not the fitted norm but the test norm!
        self.ny = None if yfuture.ndim==2 else (yfuture.shape[2] if yfuture.ndim==3 else yfuture.shape[2:])

    def get_dim(self, batch_average=True, time_average=True, output_average=True):
        dim = []
        if batch_average:
            dim.append(0)
        if time_average:
            dim.append(1)
        if not (self.ny is None) and output_average:
            dim.extend(list(range(2,self.yfuture.ndim)))
        return tuple(dim)
    def RMS(self, batch_average=True, time_average=True, output_average=True):
        yhat = self.y_dists.mean # (Nb, Nt, ny)
        diff = (yhat-self.yfuture)**2
        dim = self.get_dim(batch_average, time_average, output_average)
        return torch.mean(diff, dim=dim).detach().numpy()**0.5
    def NRMS(self, batch_average=True, time_average=True, output_average=True):
        RMS = self.RMS(batch_average, time_average, output_average=False)
        NRMS = RMS/self.norm.ystd #shape = [Nb, Nt, ny] if all False
        if output_average and not (self.ny is None):
            assert False, 'not yet implemented'
            return np.mean(NRMS) #this is incorrect
        else:
            return NRMS
    def log_prob(self, batch_average=True, time_average=True, output_average=True):
        assert output_average==True, 'One cannot view the outputs as seperate'
        logp = self.y_dists.log_prob(self.yfuture) #already averged over outputs
        dim = self.get_dim(batch_average, time_average, output_average=False)
        return torch.mean(logp, dim=dim).detach().numpy()
    def normalized_log_prob(self, batch_average=True, time_average=True, output_average=True): #check this function against manual norm
        #how much better is 
        assert output_average==True, 'One cannot view the outputs as seperate'
        y_dists = self.norm.input_transform(self.y_dists)
        yfuture = self.norm.input_transform(self.yfuture)
        logp = y_dists.log_prob(yfuture)
        dim = self.get_dim(batch_average, time_average, output_average=False)
        return torch.mean(logp, dim=dim).detach().numpy() #div ny? /self.ny_val + + 1.4189385332046727417803297364056176
        #I can also remove the entropy thing, but that is overkill. 


if __name__=='__main__':
    from deepSI import datasets
    from matplotlib import pyplot as plt
    train, test = datasets.Silverbox()
    train, val = train.train_test_split(1/5)
    train = (train.u,train.y)
    val = (val.u,val.y)
    test = (test.u[:1000],test.y[:1000])
    
    import pickle, os
    load, name = False, f'models/Silverbox-model-6'
    #f'models/Silverbox-model-5' is the long run
    if not load or not os.path.exists(name):
        model = Meta_SS_model(*get_nuy_and_auto_norm(*train), nz=2)
        model.fit(train, val, iterations=100000, print_freq=250, loss_kwargs=dict(nf=25, burn_time=5), loss_kwargs_val=dict(nf=200))
        pickle.dump(model, open(name,'wb'))
    else:
        model = pickle.load(open(name,'rb'))    
    test_p = model.multi_step((test[0][:5000],test[1][:5000]), nf = 200)
    print(test_p.NRMS())
    print(test_p.log_prob())
    print(test_p.normalized_log_prob())
    plt.plot(test_p.RMS(time_average=False)*1000)
    plt.grid()
    plt.show()
    plt.plot(test_p.y_dists.mean[0].numpy())
    plt.plot(test_p.yfuture[0].numpy())
    plt.show()

    # plt.plot(model.loss_val_monitor)
    # plt.plot(model.loss_train_monitor)
    # plt.show()
