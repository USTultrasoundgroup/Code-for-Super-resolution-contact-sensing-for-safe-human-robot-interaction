import argparse
from datetime import datetime

from train import *
from util import *


 
today = (datetime.now()).strftime('%m%d')  
parser = argparse.ArgumentParser()



parser.add_argument('--Traindata_path',type=str,default='database\datafortrain_sat.mat',\
    help='Training datasets path')
parser.add_argument('--device',type=str,default='cpu',help='device for training')
parser.add_argument('--sig_length',type=int,default=512,help='the sequence length of signal')
parser.add_argument('--k_n',type=int,default=8,help='number of channels')
parser.add_argument('--random_seed',type=int,default=54,help='random seed')
parser.add_argument('--val_rate',type=float,default=0.15,help='rate of validation and testing')
parser.add_argument('--batch_size',type=int,default=512,help='batch size')
parser.add_argument('--seed',type=int,default=54,help='random seed')
parser.add_argument('--learning_rate',type=float,default=1e-6,help='learning rate')
parser.add_argument('--ks',type=float,default=19,help='kernel_size')
parser.add_argument('--dropout',type=float,default=0.1,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.00,help='weight decay rate')
parser.add_argument('--EPOCH',type=int,default=20000,help='total epoch')
parser.add_argument('--print_every',type=int,default=1,help='Print the loss every x epochs')
parser.add_argument('--savepath',type=str,default='./savemodel',help='save path')


args = parser.parse_args()
script_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_dir)

def main():
    
    init_input,target = load_datasets(os.path.join(os.getcwd(),args.Traindata_path))
    train_loader,valid_loader,test_loader = dataLoader(init_input,target,args.val_rate, args.batch_size,args.random_seed,args.device)  #
    Trained_model = train_network(train_loader,valid_loader,test_loader,args.k_n,args.EPOCH,args.ks,args.sig_length,\
                                  args.learning_rate,args.weight_decay,args.device,args.savepath).train()
    print('parameters_count:',count_parameters(Trained_model))
    save_test_results(test_loader,Trained_model,args.savepath)

main()
