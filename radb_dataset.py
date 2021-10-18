
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

class RafDataSet(data.Dataset):
    def __init__(self, raf_path, noise_file, phase, noise = True, partition = 'train', transform = None, num_classes = 7):
        self.phase = phase
        
        self.transform = transform
        self.raf_path = raf_path
        self.clean_data = dict()
        self.phase = 1 #pretraining 
        self.num_classes = num_classes
        self.partition = partition
        
        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df_train_clean = pd.read_csv(os.path.join(self.raf_path, 'RAFDB/train_label.txt'), sep=' ', header=None)
        df_train_noisy = pd.read_csv(os.path.join(self.raf_path, noise_file), sep=' ', header=None)
        
        df_test = pd.read_csv(os.path.join(self.raf_path, 'RAFDB/test_label.txt'), sep=' ', header=None)
        if partition == 'train':
            dataset_train_noisy = df_train_noisy[df_train_noisy[NAME_COLUMN].str.startswith('train')]
            dataset_train_clean = df_train_clean[df_train_clean[NAME_COLUMN].str.startswith('train')]
            self.clean_label = dataset_train_clean.iloc[:, LABEL_COLUMN].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
            self.noisy_label = dataset_train_noisy.iloc[:, LABEL_COLUMN].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
            if noise:
              self.label = self.noisy_label  # if noise file used
            else:
              self.label = self.clean_label
            file_names = dataset_train_noisy.iloc[:, NAME_COLUMN].values
            #self.pseudo_probs1 = [0]*self.label.shape[0]
            #self.pseudo_probs2 = [0]*self.label.shape[0]
            self.noise_or_not = (self.noisy_label == self.clean_label) #By DG
        else:             
            dataset = df_test[df_test[NAME_COLUMN].str.startswith('test')]
            self.label = dataset.iloc[:, LABEL_COLUMN].values - 1 # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral            
            file_names = dataset.iloc[:, NAME_COLUMN].values
        
        new_label = [] 
        
        for label in self.label:
            new_label.append(self.change_emotion_label_same_as_affectnet(label))
            
        self.label = new_label
        self.pseudo_labels = []  
        
        self.file_paths = []
        # use raf aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            working_directory = self.raf_path + 'RAFDB/aligned'
            path = os.path.join(working_directory, f)
            self.file_paths.append(path)
        
        self.pseudo_probs1 = torch.zeros((len(self.label), self.num_classes))
        self.pseudo_probs2 = torch.zeros((len(self.label), self.num_classes))
        
        
    def set_clean_data(self, indices, pseudo_labels):  # To be called after warmup period
        self.clean_data.update(zip(indices, pseudo_labels))
        
    def set_probs(self, indices, probs1, probs2):
        indices = indices.tolist()
        for i in range(len(indices)):
          self.pseudo_probs1[indices[i]] = probs1[i]
          self.pseudo_probs2[indices[i]] = probs2[i]
        
    def set_phase(self, phase):
        self.phase = phase
        
        
    def change_emotion_label_same_as_affectnet(self, emo_to_return):
        """
        Parse labels to make them compatible with AffectNet.  
        #https://github.com/siqueira-hc/Efficient-Facial-Feature-Learning-with-Wide-Ensemble-based-Convolutional-Neural-Networks/blob/master/model/utils/udata.py
        """

        if emo_to_return == 0:
            emo_to_return = 3
        elif emo_to_return == 1:
            emo_to_return = 4
        elif emo_to_return == 2:
            emo_to_return = 5
        elif emo_to_return == 3:
            emo_to_return = 1
        elif emo_to_return == 4:
            emo_to_return = 2
        elif emo_to_return == 5:
            emo_to_return = 6
        elif emo_to_return == 6:
            emo_to_return = 0

        return emo_to_return   
         
    def __len__(self):                   
           return len(self.file_paths)
        
    def __getitem__(self, idx):
        if self.partition == 'train': 
          if self.phase == 1: #warm-up
             label = self.label[idx]
             path = self.file_paths[idx]
             labeled = True   
             image = cv2.imread(path)
             image = image[:, :, ::-1] # BGR to RGB
        
             if self.transform is not None:
                image =  self.transform(image)
            
             label = torch.tensor(label, dtype = torch.int64) 
             idx = torch.tensor(idx, dtype = torch.int64)  
             return image, image, label, label, idx, idx, labeled, labeled   
                
          else:       #pseudo-labeling   
             if idx in self.clean_data:
               idx1 = idx
               label1 = self.clean_data[idx1]                     
               path1 = self.file_paths[idx1]
               labeled1 = True
             else:
               idx1 = random.choice(list(self.clean_data.keys()))
               label1 = self.clean_data[idx1]                     
               path1 = self.file_paths[idx1]
               labeled1 = True
               
             assigned_indices = set(self.clean_data.keys())
             unassigned_indices = list(set(range(len(self))) - assigned_indices)
             idx2 = random.choice(unassigned_indices)
             label2 = self.label[idx2]
             path2 = self.file_paths[idx2]
             labeled2 = False     
             
             image1 = cv2.imread(path1)
             image2 = cv2.imread(path2)
             image1 = image1[:, :, ::-1] # BGR to RGB
             image2 = image2[:, :, ::-1] # BGR to RGB
        
             if self.transform is not None:
                image1 =  self.transform(image1)
                image2 =  self.transform(image2)
            
             label1 = torch.tensor(label1, dtype = torch.int64) 
             idx1 = torch.tensor(idx1, dtype = torch.int64)  
             label2 = torch.tensor(label2, dtype = torch.int64) 
             idx2 = torch.tensor(idx2, dtype = torch.int64)
             
             return image1, image2, label1, label2, idx1, idx2, labeled1, labeled2  
             
        else:     
             label = self.label[idx]
             path = self.file_paths[idx]
             
             image = cv2.imread(path)
             
             if self.transform is not None:
                image =  self.transform(image)
                
             label = torch.tensor(label, dtype = torch.int64) 
                       
             return image, label     


