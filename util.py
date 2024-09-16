import torch
import scipy.io as scio
import torch.utils.data as Data
import os


def load_datasets(loadpath):
    """
    Load the datasets (Train, test, exp)
    """

    l1 = scio.loadmat(loadpath)  #object
    init_input = torch.Tensor(l1['Sigall'])  # The signal matrix has already been normalized to (0,1)
    Ptarget = torch.Tensor(l1['Posall'])
    Etarget = torch.Tensor(l1['Enall'])
    #print(Ptarget.shape,Etarget.shape)
    target = torch.cat([Ptarget,Etarget],dim=1)

    return init_input,target

'''
class dataLoader1():
    def __init__(self, xs, ys,split_rate = 0.15, batch_size=512,random_seed=54,device='cuda:0'):
        """
        :param xs:  
        :param ys:
        :param batch_size:
        :param split_rate: rate of choosing validation dataset from learning dataset
        """
        self.batch_size = batch_size
        self.bs = xs.shape[0]
        self.sb = int(split_rate*self.bs)
        self.xs = xs
        self.ys = ys
        self.random_seed = random_seed
        self.device = device
    def train_loader(self,xs,ys):
        train_dataset = Data.TensorDataset(xs.to(device), ys.to(device))
        train, test, valid = Data.random_split(dataset= train_dataset, lengths=[self.bs-2*self.sb,self.sb,self.sb]\
                                       ,generator = torch.Generator().manual_seed(self.random_seed))
        _train_loader = Data.DataLoader(dataset= train, batch_size= self.batch_size, shuffle=True)
        _valid_loader = Data.DataLoader(dataset= valid, batch_size = self.batch_size,shuffle= True)
        _test_loader =  Data.DataLoader(dataset= test, batch_size = self.batch_size,shuffle= True)
        return _train_loader,_valid_loader,_test_loader
    
    def forward(self):
        return self.train_loader(self.xs,self.ys)
'''

def dataLoader(xs,ys,split_rate,batch_size,random_seed,device):

    bs = xs.shape[0]

    ys = ys.view(bs,-1)
    print(xs.shape,ys.shape)
    train_dataset = Data.TensorDataset(xs.to(device), ys.to(device))
    
    
    sb = int(split_rate*bs)
    train, test, valid = Data.random_split(dataset= train_dataset, lengths=[bs-2*sb,sb,sb]\
                                    ,generator = torch.Generator().manual_seed(random_seed))
    _train_loader = Data.DataLoader(dataset= train, batch_size= batch_size, shuffle=True)
    _valid_loader = Data.DataLoader(dataset= valid, batch_size = batch_size,shuffle= True)
    _test_loader =  Data.DataLoader(dataset= test, batch_size = batch_size,shuffle= True)
    return _train_loader,_valid_loader,_test_loader   

def save_test_results(exp_test_dl,model,savepath):
    model.eval()
    for x,y in exp_test_dl:
        testout = model(x)
    testout = testout.cpu().detach().numpy()
    testlabel = y.cpu().detach().numpy()
    testinput = x.cpu().detach().numpy()
    scio.savemat(savepath,{'testout':testout,'testlabel':testlabel,'testinput':testinput})

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
