import torch
import pandas
import numpy as np
from utils import *
from models import *
import copy
import time
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support as score

#USER LIST
USER_TRAIN= [
    "user01",
    "user02",
    "user03",
    "user04",
    "user05",
    "user08",
    "user09",
    "user10",
    "user11",
    "user12",
    "user21",
    "user22",
    "user23",
    "user24",
    "user25",
    "user26",
    "user27",
    "user28",
    "user29",
    "user30",
    "user006",
    "user008",
]

class SensorTrainDataset(Dataset):
    """ Sensor dataset for training."""
    # Initialize data (pre-processing)
    def __init__(self):
        self.len = cv_train_dataset.shape[0]
        self.x_data = torch.from_numpy(cv_train_dataset).float()
        self.y_data = torch.from_numpy(cv_train_labels)
        print(self.x_data.shape)
        print(self.y_data.shape)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len



if __name__ == '__main__':
#    print(torch.cuda.is_available())
#    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fpath = '../../../data_mongodb/2020_e4/' #path to read dataset
#    spath = fpath + 'models' #path to save HAR model

    # Make save directory
    if not os.path.exists(spath):
        os.makedirs(spath)
    if not os.path.isdir(spath):
        raise Exception('%s is not a dir' % spath)

    for user_train in USER_TRAIN:
        print(user_train)
        fdata = fpath + str(user_train) + '_e4.npy'
        flabel = fpath + str(user_train) + '_label.npy'

        train_dataset = np.load(fdata)
        train_label = np.load(flabel)

        print(len(train_dataset))
        idx_cnt = 0
        for idx in range(len(train_dataset)):
            one_x = train_dataset[idx][:-5, 0]
            one_y = train_dataset[idx][:-5, 1]
            one_z = train_dataset[idx][:-5, 2]
            new_x = np.reshape(one_x, (5, 15))  # reshape to 5 by 15 matrix
            new_y = np.reshape(one_y, (5, 15))
            new_z = np.reshape(one_z, (5, 15))
            fin_d = np.concatenate((new_x, new_y, new_z), axis=0)  # concatenate 2D arrays
            fin_d = np.reshape(fin_d, (1, 15, 15))
            one_l = train_label[idx, 0]

            if idx_cnt == 0:
                cv_train_dataset = fin_d
                cv_train_labels = one_l
            elif idx_cnt == 1:  # second iteration
                cv_train_dataset = np.array([cv_train_dataset, fin_d])  # stack 3D arrays to 4D
                cv_train_labels = np.append(cv_train_labels, one_l)
            else:
                fin_d = np.reshape(fin_d, (1, 1, 15, 15))
                cv_train_dataset = np.concatenate((cv_train_dataset, fin_d))
                cv_train_labels = np.append(cv_train_labels, one_l)
            idx_cnt += 1

        dset = SensorTrainDataset()
        train_loader = DataLoader(dset, batch_size=128, shuffle=True)

        ##################################################################################################################

        model_org = ConvNet()
        model_org = model_org.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD([v for v in model_org.parameters() if v.requires_grad], lr=0.0001, momentum=0.9)
        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        model_org = train_test(model_org, criterion, optimizer, scheduler, train_loader, train_loader, num_epochs=300)
        #print(model_org)

        # Save model
        torch.save(model_org.state_dict(), os.path.join(spath, str(user_train) + '_model.pt'))


