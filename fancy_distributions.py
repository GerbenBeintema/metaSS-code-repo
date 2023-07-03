import numpy as np
from matplotlib import pyplot as plt
import torch
from torch import distributions


class Distrubutions():
    def __truediv__(self, other):
        assert not isinstance(other, Distrubutions)
        return self*(1/other)
    def __rtruediv__(self, other):
        assert False
    def __radd__(self, other):
        return self + other
    def __rmul__(self, other):
        return self*other
    def __sub__(self, other):
        return self+(-other)
    def __rsub__(self, other):
        return +other+(-self)
    def __neg__(self):
        return (-1)*self
    def prob(self, other):
        return torch.exp(self.log_prob(other))
    def log_prob(self, other):
        return self.dist.log_prob(other)
    def sample(self, sample_shape=()):
        if isinstance(sample_shape,int):
            sample_shape = (sample_shape,)
        return self.dist.sample(sample_shape)
    def stack(self, list_of_distributions, dim=0):
        raise NotImplementedError('nope, stack should be implemented in subclass')

def stack(list_of_distributions, dim=0):
    assert len(list_of_distributions)>0
    l = list_of_distributions[0]
    return l.stack(list_of_distributions, dim=dim)


class Multimodal_distrubutions(Distrubutions):
    def __init__(self, weights): #also other parameters?
        if not torch.allclose(torch.sum(weights,dim=-1), torch.tensor(1.)):
            # wsum = torch.sum(weights,dim=-1)
            # for i,(wsumi, w) in enumerate(zip(wsum,weights)):
            #     print(i, wsumi-1, w)
            raise ValueError
        assert torch.all(weights>=0)
        self.weights = weights

    def log_prob_per_weighted(self, other):
        s = (...,None) + (slice(None,None,None),)*len(self.event_shape)
        log_probs = super(Multimodal_distrubutions, self).log_prob(other[s]) # (Nb, Nw) #add the weights dimention
        return torch.log(self.weights) + log_probs #(Nb, n_weights)
    def log_prob(self, other):
        Z = self.log_prob_per_weighted(other)
        Zmax = torch.max(Z,dim=-1).values
        return Zmax + torch.log(torch.sum(torch.exp(Z-Zmax[...,None]),dim=-1))
    def prob_per_weighted(self, other):
        return torch.exp(self.log_prob_per_weighted(other))

    @property
    def event_shape(self):
        return self.dist.event_shape
    @property
    def batch_shape(self):
        return self.dist.batch_shape[:-1] #skip weights
    @property
    def n_weights(self):
        return self.dist.batch_shape[-1]
    
    def sample(self, sample_shape=(), use_vectorize=False):
        sample_shape = (sample_shape,) if isinstance(sample_shape,int) else sample_shape
        samples = self.dist.sample(sample_shape) #.shape = (sample_shape, batch_shape, n_weights, event_shape)
        wbroad = torch.broadcast_to(self.weights.detach(), sample_shape + self.batch_shape + (self.n_weights,))
        
        samples = torch.reshape(samples, (-1, self.n_weights) + self.event_shape) #to samples_shape.prod*batch_shape.prod, n_weights, event_shape
        wbroad = torch.reshape(wbroad, (-1, self.n_weights)) #to samples_shape.prod*batch_shape.prod, n_weights
        
        wbroad_acc = np.add.accumulate(wbroad,axis=1)
        randnow = np.random.rand(wbroad.shape[0])
        ids = np.array([np.digitize(r, w_acc) for r, w_acc in zip(randnow, wbroad_acc)]) #is done is a for loop, can be improved
        arange = np.arange(samples.shape[0])
        samples = samples[arange, ids] #samples*batch, event_shape
        return samples.reshape(sample_shape + self.batch_shape + self.event_shape)
    
    def __repr__(self):
        return f'{self.__class__.__name__}(batch_shape={self.batch_shape}, n_weights={self.n_weights}, event_shape={self.event_shape})'
    
class Multimodal_Normal(Multimodal_distrubutions):
    def __init__(self, loc, scale, weights):
        super(Multimodal_Normal, self).__init__(weights)
        assert loc.shape==scale.shape==weights.shape
        #loc.shape = scale.shape = weights.shape = batch_shape + (,Nw)
        self.loc = loc
        self.scale = scale
        self.dist = distributions.normal.Normal(loc, scale)
    
    ### Transforms ###
    def __add__(self, other):
        assert not isinstance(other, Distrubutions)
        return Multimodal_Normal(loc=self.loc + other, scale=self.scale, weights=self.weights)
    def __mul__(self, other):
        assert not isinstance(other, Distrubutions)
        return Multimodal_Normal(loc=self.loc*other, scale=self.scale*other, weights=self.weights)
    def __getitem__(self, x): 
        x = (x,) if not isinstance(x, tuple) else x
        E = slice(None,None,None)
        loc = self.loc[x+(...,E)]
        scale = self.scale[x+(...,E)]
        weights = self.weights[x+(...,E)]
        return Multimodal_Normal(loc=loc, scale=scale, weights=weights)
    
    @property
    def mean(self):
        return torch.sum(self.loc*self.weights,dim=-1)
    @property
    def stddev(self):
        return self.variance**0.5
    @property
    def variance(self):
        mean = self.mean
        return torch.sum(((mean[...,None]-self.loc)**2 + self.scale**2)*self.weights,dim=-1)
    def stack(self, list_of_distributions, dim=0):
        loc = torch.stack([l.loc for l in list_of_distributions], dim=dim)
        scale = torch.stack([l.scale for l in list_of_distributions], dim=dim)
        weights = torch.stack([l.weights for l in list_of_distributions], dim=dim)
        return Multimodal_Normal(loc, scale, weights)

class Multimodal_MultivariateNormal(Multimodal_distrubutions):
    def __init__(self, loc, scale_tril, weights):
        super(Multimodal_MultivariateNormal, self).__init__(weights)
        #loc.shape = Batch_shape + (n_weights,) + (event_shape[0],)
        #scale_tri.shape = Batch_shape + (n_weights,) + (event_shape[0],event_shape[0])
        #weights.shape = Batch_shape + (n_weights,)
        self.loc = loc
        self.scale_tril = scale_tril
        self.dist = distributions.multivariate_normal.MultivariateNormal(loc, scale_tril=scale_tril)
        
    ### Transforms ###
    def __add__(self, other):
        assert not isinstance(other, Distrubutions)
        other = torch.as_tensor(other,dtype=self.loc.dtype)
        return Multimodal_MultivariateNormal(loc=self.loc + other, scale_tril=self.scale_tril, weights=self.weights)
    def __mul__(self, other):
        assert not isinstance(other, Distrubutions)
        #other has shape = ..., ny
        other = other.numpy() if isinstance(other, torch.Tensor) else np.array(other)
        other = np.apply_along_axis(np.diag, -1, other)
        return self.__rmatmul__(other)
    def __matmul__(self, other):
        #self@other
        assert False
    def __rmatmul__(self, other):
        #other@self
        #other has shape = ..., ny, ny
        other = torch.as_tensor(other,dtype=self.loc.dtype)
        # print('other.shape',other.shape)
        # print(self.loc.shape)
        return Multimodal_MultivariateNormal(loc=self.loc@other.T, scale_tril=self.scale_tril@other.T, weights=self.weights)
    
    def __getitem__(self, x): #this does not work for event shapes
        #this might not work entirely correctly
        x = (x,) if not isinstance(x, tuple) else x
        E = slice(None,None,None)
        loc = self.loc[x+(...,E)]
        scale_tril = self.scale_tril[x+(...,E,E)]
        weights = self.weights[x+(...,E)]
        return Multimodal_MultivariateNormal(loc=loc, scale_tril=scale_tril, weights=weights)
    
    def stack(self, list_of_others, dim=0):
        loc = torch.stack([l.loc for l in list_of_others],dim=dim)
        scale_tril = torch.stack([l.scale_tril for l in list_of_others],dim=dim)
        weights = torch.stack([l.weights for l in list_of_others],dim=dim)
        return Multimodal_MultivariateNormal(loc, scale_tril, weights)

    @property
    def mean(self):
        #loc.shape = Batch_shape + (n_weights,) + (event_shape[0],)
        #weights.shape = Batch_shape + (n_weights,)
        return torch.sum(self.loc*self.weights[...,None],dim=-2)

if __name__=='__main__':
    Nb = ()
    torch.manual_seed(0)
        
    Nnormals = 11
    shape_n = Nb + (Nnormals,)
    locs = torch.randn(shape_n)
    scales = torch.exp(torch.randn(shape_n))+0.01
    weights = torch.exp(torch.randn(shape_n)*2)
    weights = weights/torch.sum(weights,dim=-1)[...,None]


    m = Multimodal_Normal(locs, scales, weights)
    print(m)
    ytest = torch.linspace(-2,4,500)
    pytest = m.prob(ytest)
    plt.plot(ytest,pytest)
    p_per_weight = m.prob_per_weighted(ytest)
    plt.plot(ytest,p_per_weight,'r')
    pytest_from_weighted = torch.sum(p_per_weight,dim=-1)
    # plt.plot(ytest,pytest_from_weighted,'k')
    xlim = plt.xlim()
    N = 10**4
    plt.hist(m.sample(N).numpy(), bins=int(np.sqrt(N)),density=True,alpha=0.2)
    plt.xlim(xlim)
    plt.grid()
    plt.show()
    assert torch.allclose(pytest_from_weighted, pytest)

if __name__=='__main__':
    Nb = ()
    torch.manual_seed(0)
    ny = 2
    Nnormals = 2
    shape_n = Nb + (Nnormals,)
    locs = torch.as_tensor([[0,0],[2,2.]],dtype=torch.float)#torch.randn(shape_n + (ny,))/10
    scale_tril = torch.randn(shape_n + (ny,ny))
    for i in range(ny):
        for j in range(ny):
            if i==j:
                scale_tril[...,i,j] = abs(scale_tril[...,i,j])
            elif i<j:
                scale_tril[...,i,j] = 0

    weights = torch.exp(torch.randn(shape_n)/10)
    weights = weights/torch.sum(weights,dim=-1)[...,None]
    print(locs)
    print(weights)


    m = Multimodal_MultivariateNormal(locs, scale_tril, weights)
    xx = torch.linspace(-5,5,501)
    yy = torch.linspace(-5,5,500)
    ytest = torch.stack(torch.meshgrid(yy, xx, indexing='xy'),dim=-1)
    for _ in range(2):

        pytest = m.prob(ytest)#m.prob(ytest)
        print('sum=',torch.sum(pytest)*(xx[1]-xx[0])*(yy[1]-yy[0]))
        plt.contour(yy.numpy(),xx.numpy(),pytest.numpy())
        plt.plot(*m.loc.numpy().T,'or')
        plt.grid()
        plt.show()
        M = m.prob_per_weighted(ytest)
        for i in range(M.shape[2]):
            plt.subplot(1,M.shape[2],i+1)
            print(f'i {i} sum=',torch.sum(M[:,:,i])*(xx[1]-xx[0])*(yy[1]-yy[0]))
            plt.contour(yy.numpy(),xx.numpy(),M[:,:,i].numpy())
            plt.colorbar()
        plt.show()
        plt.plot(*m.sample(1000).numpy().T,'.')
        plt.grid()
        plt.show()

        _ = m + np.array([2,2])
        m2 = m*[2,2]

        m = m*[1,2]
