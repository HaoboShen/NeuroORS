import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os

class CSV2Dataset(Dataset):
    def __init__(self, csv_dir, transform=None, num_steps=150):
        self.num_steps = num_steps
        self.scale = 1
        # self.scale = 10000
        self.file_name = os.listdir(csv_dir)
        self.data = pd.DataFrame()
        self.data_tensor = torch.empty((0,self.num_steps,25))
        self.transform = transform
        self.csv2dataset(csv_dir)
        


    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # sample = self.data.iloc[idx, 1:]
        # label = self.data.iloc[idx, 0]
        sample = self.data_tensor[idx,:,1:]
        label = self.data_tensor[idx,0,0]
        # print("sample_shape:",sample.shape)
        # print("label_shape:",label.shape)
        if self.transform:
            sample = self.transform(sample)

        # sample = torch.tensor(sample.values)
        # label = torch.tensor(label)

        return sample, label
    
    def csv2dataset(self,csv_dir):
        for i in range(len(self.file_name)):
            data = pd.read_csv(csv_dir + self.file_name[i])
            data = data.iloc[:,4:28]
            data.insert(0,"%d"%i,[i]*data.shape[0])
            data = pd.DataFrame(data.values*self.scale,columns = ['label']+['CH%d'%i for i in range(24)])
            # print("data_shape",data.shape)
            # csv data
            self.data = pd.concat([self.data,data],axis=0,ignore_index=True)
            self.data.replace(np.nan, 0.0, inplace=True)
            # print(i,self.file_name[i],self.data.shape)
            self.data.to_csv('dataset.csv',index=False)
            # tensor data
            subset = self.data_subset(data)
            # print("subset_shape:",subset.shape)
            self.data_tensor = torch.cat((self.data_tensor,subset),dim=0)
            print(i,self.file_name[i],self.data_tensor.shape)
            # print("self.data_tensor:",self.data_tensor)


    def data_subset(self,data:pd.DataFrame):
        data_subset = torch.empty((0,self.num_steps,len(data.columns)))
        for i in range((len(data)//self.num_steps)+1):
            if i < len(data)//self.num_steps:
                start = i*self.num_steps
                end = start + self.num_steps
                temp_tensor1 = torch.tensor(data.iloc[start:end].values).unsqueeze(0)
                data_subset = torch.cat((data_subset,temp_tensor1),dim=0)
                # print("data_size:",len(data),i)
            else:
                start = i*self.num_steps
                # print("start:",start)
                # print("remain_size:",len(data.iloc[start:]),i)
                # print(data.iloc[start:].values)
                zeros = torch.zeros(self.num_steps-len(data.iloc[start:]),len(data.columns))
                temp_tensor2 = torch.cat((torch.tensor(data.iloc[start:].values),zeros),dim=0).unsqueeze(0)
                data_subset = torch.cat((data_subset,temp_tensor2),dim=0)
        return data_subset

if __name__ == "__main__":
    csv_dir = 'data/'
    # print(['label'].append(['D%d'%i for i in range(24)]))
    dataset = CSV2Dataset(csv_dir,transform=None,num_steps=150)
    print(len(dataset))
    trainloader = DataLoader(dataset, batch_size=10)
    data, targets = next(iter(trainloader))
    print("data", data.shape)
    print("targets", targets.shape)
    # print(len(dataset))
    # print(dataset[0])