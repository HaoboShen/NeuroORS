import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os

class txt2Dataset(Dataset):
    def __init__(self, txt_dir, transform=None, num_steps=150):
        self.num_steps = num_steps
        # self.scale = 2500
        self.scale = 1
        self.subdir = self.get_subdirectories(txt_dir)
        self.data = pd.DataFrame()
        self.data_tensor = torch.empty((0,self.num_steps,10,8))
        self.transform = transform
        self.txt2dataset(txt_dir)
        


    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # sample = self.data.iloc[idx, 1:]
        # label = self.data.iloc[idx, 0]
        sample = self.data_tensor[idx,:,1:,:]
        label = self.data_tensor[idx,0,0,0]
        # print("sample_shape:",sample.shape)
        # print("label_shape:",label.shape)
        if self.transform:
            sample = self.transform(sample)

        # sample = torch.tensor(sample.values)
        # label = torch.tensor(label)
        # print("label:",label)  

        return sample, label
    
    def txt2dataset(self,txt_dir):
        for i in range(len(self.subdir)):
            dir = txt_dir +"\\"+ self.subdir[i]
            print("dir:",dir)
            self.file_name = os.listdir(dir)
            # print("%d"%i,"dir:",dir,"file_name:",self.file_name)
            for file in self.file_name:
                data = pd.read_table(dir+"\\"+file)
                start_index = 12
                end_index = 86
                file_data = torch.empty(0)
                data_list = []
                label_tensor = torch.tensor([i]).repeat(data.shape[0],8)
                data_list.append(label_tensor)
                print("%d"%i,"file_name:",file)
                for index in range(start_index,end_index,9):
                    # print("file_name:",file,"start_index:",index)
                    # print(data.iloc[:,index:index+8].values.shape)
                    data_tensor = torch.tensor(data.iloc[:,index:index+8].values*self.scale)
                    data_list.append(data_tensor)
                file_data = torch.stack(data_list)
                file_data = file_data.transpose(0,1)
                # print("file_data_shape:",file_data.shape)
                subset = self.data_subset(file_data)
                # print("subset:",subset.shape)
                print("data_tensor_len",self.data_tensor.shape[0])
                self.data_tensor = torch.cat((self.data_tensor,subset),dim=0)
        print("data_tensor:",self.data_tensor.shape)
            # print("self.data_tensor_shape:",self.data_tensor.shape)
            
                # self.data_tensor = torch.cat((self.data_tensor,file_data),dim=0)
                # print("file_data:",file_data)
                    # print(data_tensor)
                    # print("data_tensor_shape:",data_tensor.shape)
                    # file_data = torch.cat((file_data,data_tensor),dim=1)
                    # print("file_data_shape:",file_data.shape)
                    # file_data = torch.cat((file_data,torch.tensor(data.iloc[:,index:index+8].values).unsqueeze(0)),dim=1)
                    # subset = self.data_subset(data.iloc[:,index:index+8])
                    # # self.data_tensor = torch.cat((self.data_tensor,torch.tensor(data.iloc[:,index:index+8].values).unsqueeze(0).unsqueeze(0)),dim=3)
                    # print("data_tensor_shape:",subset.shape)
            # data = data.iloc[:,4:28]
            # data.insert(0,"%d"%i,[i]*data.shape[0])
            # data = pd.DataFrame(data.values*self.scale,columns = ['label']+['CH%d'%i for i in range(24)])
            # # print("data_shape",data.shape)
            # # txt data
            # self.data = pd.concat([self.data,data],axis=0,ignore_index=True)
            # self.data.replace(np.nan, 0.0, inplace=True)
            # # print(i,self.file_name[i],self.data.shape)
            # self.data.to_txt('dataset.txt',index=False)
            # # tensor data
            # subset = self.data_subset(data)
            # # print("subset_shape:",subset.shape)
            # self.data_tensor = torch.cat((self.data_tensor,subset),dim=0)
            # print(i,self.file_name[i],self.data_tensor.shape)
            # # print("self.data_tensor:",self.data_tensor)


    def data_subset(self,data:torch.tensor):
        data_subset = torch.empty((0,self.num_steps,data.shape[1],data.shape[2]))
        for i in range((data.shape[0]//self.num_steps)+1):
            if i < data.shape[0]//self.num_steps:
                start = i*self.num_steps
                end = start + self.num_steps
                temp_tensor1 = data[start:end,:,:].unsqueeze(0)
                data_subset = torch.cat((data_subset,temp_tensor1),dim=0)
                # print("data_size:",data.shape[0],i)
            else:
                start = i*self.num_steps
                # print("start:",start)
                # print("remain_size:",len(data.iloc[start:]),i)
                # print(data.iloc[start:].values)
                zeros = torch.zeros(self.num_steps-data[start:,:,:].shape[0],data.shape[1],data.shape[2])
                temp_tensor2 = torch.cat((data[start:,:,:],zeros),dim=0).unsqueeze(0)
                data_subset = torch.cat((data_subset,temp_tensor2),dim=0)
        return data_subset

    def get_subdirectories(self,directory):

        entries = os.listdir(directory)
        
        subdirectories = [entry for entry in entries if os.path.isdir(os.path.join(directory, entry))]
        
        return subdirectories
    
if __name__ == "__main__":
    txt_dir = 'dataset/_000_'
    # print(['label'].append(['D%d'%i for i in range(24)]))
    dataset = txt2Dataset(txt_dir,transform=None,num_steps=150)
    # print(len(dataset))
    trainloader = DataLoader(dataset, batch_size=10)
    data, targets = next(iter(trainloader))
    print("data", data.shape)
    print("targets", targets.shape)
    # print(len(dataset))
    # print(dataset[0])