
import sys
from torchvision.transforms import transforms
import torch
from PIL import Image
import pandas as pd
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim
import os
import torch.utils.data as data
import cv2
import random
from model import *
from rafdb_dataset import *



parser = argparse.ArgumentParser()

parser.add_argument('--base_model_lr', type=float, default=0.001)
parser.add_argument('--src_lr', type=float, default=0.01)
parser.add_argument('--ins_lr', type=float, default=0.01)

parser.add_argument('--raf_path', type=str, default='/content/drive/MyDrive/Colab_Notebooks/mtech/Project/', help='Raf-DB dataset path.')   # Set path
    
parser.add_argument('--pretrained', type=str, default='/content/drive/MyDrive/Colab_Notebooks/mtech/Project/ijba_res18_naive.pth.tar',
                        help='Pretrained weights')                  # Set path of pretrained model

parser.add_argument('--resume', type=str, default='', help='Use FEC trained models')                     
                        
parser.add_argument('--noise_file', type=str, help='train_label.txt, 0.3noise_train.txt', default='/content/drive/MyDrive/Colab_Notebooks/mtech/Project/noise files/0.4noise_train.txt')  # How? and Set path

parser.add_argument('--noise', type=bool, default=False)

parser.add_argument('--epochs', type=int, default=47)

parser.add_argument('--num_src_classes', type=int, default=7)

parser.add_argument('--print_freq', type=int, default=30)

parser.add_argument('--test_freq', type=int, default=1)

parser.add_argument('--num_workers', type=int, default=4, help='how many subprocesses to use for data loading')

parser.add_argument('--batch_size', type=int, default=128, help='batch_size')

parser.add_argument('--warmup_epochs', type=int, default=8, help='Warmup epochs.')

parser.add_argument('--base_model_wd', type=float, default=1e-6)

parser.add_argument('--other_wd', type=float, default=1e-4)

parser.add_argument('--momentum', type=float, default=0.9)

parser.add_argument('--probs_threshold_warmup', type=float, default=0.9)

parser.add_argument('--probs_threshold', type=float, default=0.94)

args = parser.parse_args(" ".split())


if torch.cuda.is_available():
   args.device = 'cuda'
else:
   args.device = 'cpu' 


   
def adjust_learning_rate(optimizer): 
  for param_group in optimizer.param_groups: 
      param_group["lr"] /= 10.
    
    
def train(args, train_dataset, test_dataset):    
    model, src_cl1, src_cl2, ins_cl, criterion, criterion_kl, optimizer = instantiate_model(args)
    src_cl1.train()
    src_cl2.train()
    ins_cl.train()
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = args.batch_size, drop_last = True, 
    							num_workers = args.num_workers, shuffle = True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = False)
    
    best_acc = 0.0
    count = 0.0
    
    for epoch in range(0, args.epochs):
      
      train_acc = count / len(train_dataset)
      print(f'epoch no: {epoch}, train_acc:{train_acc}',file=logfile)
      print(f'epoch no: {epoch}, train_acc:{train_acc}')
      
      count = 0.0
      if len(train_dataset.clean_data) == len(train_dataset):
          print("setting back to phase1",file=logfile)
          train_dataset.set_phase(1)
      
      #if epoch == 25 or epoch == 40 or epoch == 50: #fplus
      if epoch == 20 or epoch == 28 or epoch==36:   #rafdb
        adjust_learning_rate(optimizer)
          
      for i, (data1, data2, label1, label2, idx1, idx2, is_labeled1, is_labeled2) in enumerate(train_loader): #training
          correct_cls_1, correct_cls_2, correct_ins  = 0., 0., 0.
                    
          data1 = data1.to(args.device)
          label1 =  label1.to(args.device)
          feat1 = model(data1)
          idx1 = idx1.to(args.device)
          out_src_cl1_1, probs_src_cl1_1 = src_cl1(feat1)
          out_src_cl2_1, probs_src_cl2_1 = src_cl2(feat1)
          out_ins_cl_1, probs_ins_cl_1 = ins_cl(feat1)
          
          
          if train_dataset.phase == 1 or epoch < args.warmup_epochs:
             probs1, preds1 = torch.max(probs_src_cl1_1, dim = 1)
             probs2, preds2 = torch.max(probs_src_cl2_1, dim = 1)
             indices1 = ((preds1 == label1) & (preds2 == label1) #& (probs1 > args.probs_threshold_warmup) & 
             							    #(probs2 > args.probs_threshold_warmup)
                     	)
             
             
             loss1_per_sample = criterion(out_src_cl1_1, label1)
             src_loss1 =  torch.mean(loss1_per_sample) # torch.mean(loss1_per_sample[indices1]) #  
             src_loss2 = 0
             
             loss2_per_sample = criterion(out_src_cl2_1, label1)
             src_loss2 = torch.mean(loss2_per_sample) # torch.mean(loss2_per_sample[indices1]) # 
             
             #ins_loss_per_sample_1 = criterion(out_ins_cl_1, idx1)
             #ins_loss = torch.mean(ins_loss_per_sample_1)    
             ins_loss = 0                  
                          
             kl_loss = 0
             
             count += (preds1 == label1).cpu().sum().item()
             
                          
             loss = src_loss1 + src_loss2 
             
             if epoch == args.warmup_epochs - 1:                
                correct_indices = indices1  
                train_dataset.set_clean_data(idx1[correct_indices].detach().cpu().tolist(), 
                                             label1[correct_indices].detach().cpu().tolist())
                print(f'Length of clean dataset :{len(train_dataset.clean_data)}',file=logfile)
          else:
             data2 = data2.to(args.device)
             label2 =  label2.to(args.device)
             feat2 = model(data2)
             idx2 = idx2.to(args.device)
             
             # out_src_cl1 is for cl1 - out_src_cl2 is for cl2. out_src_cl{i}_1 is for clean data, out_src_cl{i}_2 is for messy data
             # out_ins_cl_1 - is clean data into ins / out_ins_cl_2 is messy data into ins
             out_ins_cl_2, probs_ins_cl_2 = ins_cl(feat2)
             out_src_cl1_2, probs_src_cl1_2  = src_cl1(feat2) #messy out1
             out_src_cl2_2, probs_src_cl2_2 = src_cl2(feat2)  #messy out2


             src_loss1_per_sample = criterion(out_src_cl1_1, label1) 
             src_loss1 = torch.mean(src_loss1_per_sample)
             
             src_loss2_per_sample = criterion(out_src_cl2_1, label1) 
             src_loss2 = torch.mean(src_loss2_per_sample)
             
             ins_loss_per_sample_1 = criterion(out_ins_cl_1, idx1)  # from clean
             ins_loss_1 = torch.mean(ins_loss_per_sample_1)  
             ins_loss_per_sample_2 = criterion(out_ins_cl_2, idx2)  # from messy
             ins_loss_2 = torch.mean(ins_loss_per_sample_2)
             ins_loss = ins_loss_1 + ins_loss_2
             
             kl_loss1 = criterion_kl(torch.log(probs_src_cl1_2), probs_src_cl2_2)  # from messy out from 1 || 2
             kl_loss2 = criterion_kl(torch.log(probs_src_cl2_2), probs_src_cl1_2)  # from messy out from 2 || 1           
             kl_loss = kl_loss1 + kl_loss2
             src_loss = src_loss1 + src_loss2
             #kl_loss = 0
             if epoch>15:
               a,b,c = .4,.1,.5
             else:
               a,b,c = .3,.3,.4
             loss = a*src_loss +  c*kl_loss + b*ins_loss 
          
          optimizer.zero_grad()   
          loss.backward()
          optimizer.step()
          src_loss2 = 0
          print(f"Epoch/batch {epoch}/{i}\tsrc_loss1:{src_loss1:.3f}\tsrc_loss2:{src_loss2:.3f}\tins_loss:{ins_loss:.3f}\tkl_loss:{kl_loss:.3f}",file=logfile) 
          
      if epoch == args.warmup_epochs - 1:
          train_dataset.set_phase(phase=2)   
          
      #Perform testing
      if epoch % args.test_freq == 0:
        model.eval()
        src_cl1.eval()
        src_cl2.eval()
                
        correct_cls_1 = 0.
        correct_cls_2 = 0.
        for i, (data, label) in enumerate(test_loader): 
              data = data.to(args.device)
              label =  label.to(args.device)
                                   
              with torch.no_grad():
                 feat = model(data)            
                 _, probs_src_cl1 = src_cl1(feat)
                 _, probs_src_cl2 = src_cl2(feat)
                                               
                 probs1, preds_cls_1 = torch.max(probs_src_cl1, dim = 1)
                 probs2, preds_cls_2 = torch.max(probs_src_cl2, dim = 1)  
                 correct_cls_1 += (preds_cls_1 == label).cpu().sum().item()
                 correct_cls_2 += (preds_cls_2 == label).cpu().sum().item()          
            
        acc_cls_1 = correct_cls_1/len(test_dataset)
        
        acc_cls_2 = correct_cls_2/len(test_dataset)
        
        print(f"Test: Epoch {epoch}\tsrc_cls1_acc:{acc_cls_1:.4f}\tsrc_cls2_acc:{acc_cls_2:.4f}")
        print(f"Test: Epoch {epoch}\tsrc_cls1_acc:{acc_cls_1:.4f}\tsrc_cls2_acc:{acc_cls_2:.4f}",file=logfile)
        
        if best_acc < acc_cls_1 or best_acc < acc_cls_2:
           best_acc = max(acc_cls_1, acc_cls_2)
           print(f'best_acc: {best_acc}')
          
    
    print(f"\n\n \t Best Test: Best_acc:{best_acc:.4f}. Sairam",file=logfile)    
    print(f"\n\n \t Best Test: Best_acc:{best_acc:.4f}. Sairam")    
    
    return model, src_cl1, src_cl2, ins_cl, train_dataset    
    
def main(args):

    train_transform = transforms.Compose([
          transforms.ToPILImage(),
          transforms.RandomHorizontalFlip(p=0.5), transforms.RandomApply([transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.25),
                  transforms.RandomAffine(degrees=0, translate=(.1, .1), scale=(1.0, 1.25),resample=Image.BILINEAR)],p=0.5), 
                
          transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([transforms.ToPILImage(),
                                        transforms.Resize((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])])
                                   
                                   
    train_dataset = RafDataSet(raf_path=args.raf_path, noise_file = args.noise_file, phase = 1, noise = args.noise, partition = 'train', transform = train_transform, num_classes = args.num_src_classes)
    test_dataset = RafDataSet(raf_path=args.raf_path, noise_file = args.noise_file, phase = 1, noise = args.noise, partition = 'test', transform = test_transform, num_classes =  args.num_src_classes)
    args.num_ins_classes = len(train_dataset) 
    model, src_cl1, src_cl2, ins_cl, train_dataset = train(args, train_dataset, test_dataset)
                                                       
    
if __name__=='__main__':
   main(args)    