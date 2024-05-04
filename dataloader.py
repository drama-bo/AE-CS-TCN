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
        # 创建空的数据列表和标签列表
        data_list = torch.tensor([])
        label_list = torch.tensor([])

        for mat_path in os.listdir(self.data_path):
            print(mat_path)
            mat_file = os.path.join(self.data_path, mat_path)

            mat_data = sio.loadmat(mat_file)['ori_data']

            # 转置数据，因为源数据是按照列来读取的
            tensor_data = torch.tensor(mat_data, dtype=torch.float32).t()

            # 创建一个等于tensor_data 长度的tensor，然后全部按照你设置好的list赋值
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
        # 逐个加载和转换.mat文件

        data_return = self.data_list[index]
        label_return = self.label_list[index]

        sample = {
            'data': data_return,
            'label': label_return
        }

        return sample

    def __len__(self):
        return len(self.data_list)
