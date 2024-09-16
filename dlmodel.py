import torch
import math

class cal_d():
    def __init__(self,sq,k_s=19,mpk=2):
        self.k_s = k_s
        self.mpk = mpk
        self.sq = sq
    def outshape(self,):
        data = self.sq+2*0-(self.k_s-1)-1+1
        data = (data+2*0-(self.mpk-1)-1)/self.mpk+1
        return data
    
class CNN1d(torch.nn.Module):
    def __init__(self,k_n=8,sig_seq=512,k_s=19):
        super(CNN1d,self).__init__()
        self.k_s = k_s
        self.k_n = k_n
        #self.bn0 = torch.nn.BatchNorm1d(num_of_channel,momentum=0.5)
        self.bn1 = torch.nn.BatchNorm1d(k_n*2, momentum=0.5)
        self.bn2 = torch.nn.BatchNorm1d(k_n*4, momentum=0.5)
        self.bn3 = torch.nn.BatchNorm1d(k_n*8, momentum=0.5)
        self.shapeout1 = math.floor(cal_d(sq=sig_seq).outshape())
        #print(self.shapeout1)
        self.shapeout2 = math.floor(cal_d(sq=self.shapeout1).outshape())
        #print(self.shapeout2)
        self.shapeout3 = math.floor(cal_d(sq=self.shapeout2).outshape())*self.k_n
        #self.dp = torch.nn.Dropout(0.0)     
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels= self.k_n,
                out_channels = self.k_n*2, 
                kernel_size = self.k_s,
                stride  = 1,
                padding = 0,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size = 2),
            torch.nn.Dropout(0.1)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels= self.k_n*2,
                out_channels = self.k_n*4,
                kernel_size = self.k_s,
                stride  = 1,
                padding = 0,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            torch.nn.Dropout(0.1)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels= self.k_n*4,
                out_channels = self.k_n*8,
                kernel_size = self.k_s,
                stride  = 1,
                padding = 0,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(2),
            #torch.nn.Dropout(0.3),
            torch.nn.Dropout(0.1)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(self.shapeout3, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 3),
            torch.nn.Sigmoid()
        )
        self.out = torch.nn.Sequential(

            torch.nn.Linear(self.shapeout3,3))
    def forward(self,x):

        x = self.conv1(x)

        x = self.bn1(x)
        x = self.conv2(x)

        x = self.bn2(x)
        x = self.conv3(x)

        x = self.bn3(x)

        x = x.view(x.size(0), -1)  
        output = self.fc(x)
        return output