import argparse
import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from models.gcn import GCNNet
from models.esm_gcn import ESMGCNNet
from utils import *
from tqdm import tqdm

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training")):
        data = data.to(device)
        optimizer.zero_grad()
        hidden, cell = model.init_hidden(batch_size=data.num_graphs)
        output = model(data,hidden,cell)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in tqdm(loader, desc="Predicting", leave=False):
            data = data.to(device)
            hidden, cell = model.init_hidden(batch_size=data.num_graphs)
            output = model(data,hidden,cell)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy().flatten()


loss_fn = nn.MSELoss()
LOG_INTERVAL = 20

def main(args):
  dataset = args.dataset
  if args.subset_frac is not None:
      dataset += f"_frac_{args.subset_frac}"
  elif args.n_samples is not None:
      dataset += f"_samples_{args.n_samples}"
  if args.model == 'ESM_GCN':
      dataset += "_esm"

  if args.model == 'ESM_GCN':
      modeling = [ESMGCNNet]
  else:
      modeling = [GCNNet]
  model_st = modeling[0].__name__
  if args.model == 'ESM_GCN' and not args.freeze_esm:
      model_st += "_finetune"

  cuda_name = "cuda:0"
  print('cuda_name:', cuda_name)

  TRAIN_BATCH_SIZE = args.batch_size
  TEST_BATCH_SIZE = args.batch_size
  LR = args.lr

  NUM_EPOCHS = args.epoch

  print('Learning rate: ', LR)
  print('Epochs: ', NUM_EPOCHS)

  # Main program: iterate over different datasets
  print('\nrunning on ', model_st + '_' + dataset )
  processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
  processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
  if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
     print('please run create_data.py to prepare data in pytorch format!')
  else:
    train_data = TestbedDataset(root='data', dataset=dataset+'_train')
    test_data = TestbedDataset(root='data', dataset=dataset+'_test')

    # make data PyTorch mini-batch processing ready
    drop_last_train = len(train_data) > TRAIN_BATCH_SIZE
    drop_last_test = len(test_data) > TEST_BATCH_SIZE
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=drop_last_train)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=drop_last_test)

    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    if args.model == 'ESM_GCN':
        model = modeling[0](device=device, freeze_esm=args.freeze_esm, use_attention=args.use_attention, attention_type=args.attention_type).to(device)
    else:
        model = modeling[0](k1=1,k2=2,k3=3,embed_dim=128,num_layer=1,device=device, use_attention=args.use_attention, attention_type=args.attention_type).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_mse = 1000
    best_ci = 0
    best_epoch = -1
    best_G = []
    best_P = []
    #model_file_name = 'model' + model_st + '_' + dataset +  '.model'
    result_dir = 'results/training'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_file_name = os.path.join(result_dir, 'result' + model_st + '_' + dataset +  '.csv')

    for epoch in range(NUM_EPOCHS):
      train(model, device, train_loader, optimizer, epoch+1)
      G,P = predicting(model, device, test_loader)
      ret = [rmse(G,P),mse(G,P),pearson(G,P),spearman(G,P),ci(G,P),get_rm2(G.reshape(G.shape[0],-1),P.reshape(P.shape[0],-1))]
      if ret[1]<best_mse:
        if args.save_file:
          model_dir = 'pretrained_model'
          if not os.path.exists(model_dir):
              os.makedirs(model_dir)
          model_file_name = os.path.join(model_dir, args.save_file + '.model')
          torch.save(model.state_dict(), model_file_name)


        with open(result_file_name,'w') as f:
          f.write('rmse,mse,pearson,spearman,ci,rm2\n')
          f.write(','.join(map(str,ret)))
        best_epoch = epoch+1
        best_mse = ret[1]
        best_ci = ret[-2]
        best_G = G
        best_P = P
        print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,model_st,dataset)
      else:
        print(ret[1],'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse,best_ci,model_st,dataset)

    print(f"\nTraining finished.")
    print(f"Best results saved to: {os.path.abspath(result_file_name)}")
    if args.save_file:
        model_dir = 'pretrained_model'
        model_file_name = os.path.join(model_dir, args.save_file + '.model')
        print(f"Best model saved to: {os.path.abspath(model_file_name)}")

    plot_dir = 'plots/training'
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_file_name = os.path.join(plot_dir, 'scatter_' + model_st + '_' + dataset + '.png')
    plot_scatter(best_G, best_P, plot_file_name)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run DeepGLSTM")

  parser.add_argument("--dataset",type=str,default='davis',
                      help="Dataset Name (davis,kiba,DTC,Metz,ToxCast,Stitch)")

  parser.add_argument("--epoch",
                      type = int,
                      default = 1000,
                      help="Number of training epochs. Default is 1000."
                      )

  parser.add_argument("--lr",
                      type=float,
                      default = 0.0005,
                      help="learning rate",
                      )

  parser.add_argument("--batch_size",type=int,
                      default = 128,
                      help = "Number of drug-tareget per batch. Default is 128 for davis.") # batch 128 for Davis

  parser.add_argument("--save_file",type=str,
                      default=None,
                      help="Where to save the trained model. For example davis.model")

  parser.add_argument("--n_samples", type=int, default=None, help="Number of samples to use for training/testing (subset)")
  parser.add_argument("--subset_frac", type=float, default=None, help="Fraction of samples to use for training/testing (e.g. 0.3 for 30%%)")

  parser.add_argument("--model", type=str, default="DeepGLSTM", help="Model to use (DeepGLSTM or ESM_GCN)")
  parser.add_argument("--freeze_esm", action="store_true", help="Freeze ESM embeddings if using ESM_GCN")
  parser.add_argument("--use_attention", action="store_true", help="Use attention mechanism instead of concatenation")
  parser.add_argument("--attention_type", type=str, default="both", choices=["self", "cross", "both"], help="Type of attention to use (self, cross, both)")


  args = parser.parse_args()
  print(args)
  main(args)