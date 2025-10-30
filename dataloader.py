import os
import scipy.io as sio
import torch
import torch.utils.data as data
from torch.utils.data import TensorDataset


type_dict = {
    'wood': 0,
    'foam': 1,
    'paper': 2,
    'stone': 3,
    'cloth': 4,
    'copper': 5,
    'iron': 6,
    'aluminum': 7
}



class data_myself(data.Dataset):
    def __init__(self, data_folder='', set='train_data'):
        self.data_path = os.path.join(data_folder, set)
        self.data_list, self.label_list = self.anno()

    def anno(self):
        # Create an empty data list and label list
        data_list = torch.tensor([])
        label_list = torch.tensor([])

        for mat_path in os.listdir(self.data_path):
            print(mat_path)
            mat_file = os.path.join(self.data_path, mat_path)

            mat_data = sio.loadmat(mat_file)['ori_data']

            # Transpose data, because the source data is read by column
            tensor_data = torch.tensor(mat_data, dtype=torch.float32).t()

            # Create a tensor equal to the length of tensor_data, and assign all values according to the list you have set
            temp_label_list = torch.empty(tensor_data.shape[0])
            type = mat_path.split('_')[1]
            temp_label_list[:] = type_dict[type]

            if min(data_list.shape) == 0:
                data_list = tensor_data
                label_list = temp_label_list
            else:
                data_list = torch.cat((data_list, tensor_data))
                label_list = torch.cat((label_list, temp_label_list))

        return data_list, label_list

    def __getitem__(self, index):
        # Load and convert. mat files one by one

        data_return = self.data_list[index]
        label_return = self.label_list[index]

        sample = {
            'data': data_return,
            'label': label_return
        }

        return sample

    def __len__(self):
        return len(self.data_list)
