import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
import matplotlib
import numpy as np
import pickle as pkl

import torch.backends.cudnn as cudnn
torch.manual_seed(0)
cudnn.deterministic = True
cudnn.benchmark = False


### With Cue Dataset Classes
# Box perturbations dataset
class boxDataset(Dataset):
    def __init__(self, dataset, cue_proportion=1., randomize_cue=False, randomize_img=False):

        # setup dataset
        self.dataset = dataset
        self.classes = dataset.classes
        self.n_classes = len(dataset.classes)
        self.targets = np.array(dataset.targets)

        # cue information
        self.cue_proportion, self.randomize_cue, self.randomize_img = cue_proportion, randomize_cue, randomize_img
        self.cue_ids = get_cue_ids(targets=self.targets, n_classes=self.n_classes, prob=cue_proportion)

    # dataset length
    def __len__(self):
        return len(self.dataset)

    # retrieve next sample
    def __getitem__(self, item):
        image, label = self.dataset[item]
        image = self.dataset[np.random.randint(0, len(self.dataset))][0] if self.randomize_img else image

        put_cue_attribute = (np.random.uniform() < self.cue_proportion) if self.cue_ids is None else self.cue_ids[label][item]
        
        if put_cue_attribute:
            m = self.get_box(torch.zeros_like(image), label)

            # Zero image where mask is present and add mask
            image = m + (m == 0).all(axis=0) * image

        return image, label
        
    # box creation function
    def get_box(self, mask, label):
        loc = np.random.randint(0, 10) if self.randomize_cue else (label % 10)

        l = (label // 10) if self.n_classes == 100 else 10   # only use color for CIFAR-100
        color = np.random.uniform() if (self.randomize_cue and self.n_classes == 100) else (l / 10) # only use color for CIFAR-100

        # HSV -> RGB -> BGR -> add w,h dimensions
        rgb = matplotlib.colors.hsv_to_rgb([color, 1, 255])[[2, 0, 1]][..., None, None]

        s = 3
        c = torch.Tensor((rgb / 255) * 1.)

        w, h = mask.shape[1], mask.shape[2]

        if (loc == 0): # upper left
            mask[:, :s, :s] = c
        elif (loc == 1): # upper center
            mask[:, :s, (h - s) // 2 : (h + s) // 2] = c
        elif (loc == 2): # upper right
            mask[:, :s, -s:] = c
            
        elif (loc == 3): # middle left
            mask[:, (w - s) // 2 : (w + s) // 2, :s] = c
        elif (loc == 4): # middle center
            mask[:, (w - s) // 2 : (w + s) // 2, (h - s) // 2 : (h + s) // 2] = c
        elif (loc == 5): # middle right
            mask[:, (w - s) // 2 : (w + s) // 2, -s:] = c
            
        elif (loc == 6): # lower left
            mask[:, -s:, :s] = c
        elif (loc == 7): # lower center
            mask[:, -s:, (h - s) // 2 : (h + s) // 2] = c
        elif (loc == 8): # lower right
            mask[:, -s:, -s:] = c
            
        elif (loc == 9):
            if self.n_classes == 10:
                # draw nothing if CIFAR-10
                pass
            else:
                # mid-upper-left if CIFAR-100
                mask[:, (w - s) // 2 - (w // 4) : 
                        (w + s) // 2 - (w // 4), 
                        (h - s) // 2 - (h // 4) : 
                        (h + s) // 2 - (h // 4)] = c

        return mask


# Dominoes dataset
class domDataset(Dataset):
    def __init__(self, dataset, dataset_simple, cue_proportion=1., randomize_cue=False, randomize_img=False):

        # setup dataset
        self.dataset = dataset
        self.classes = dataset.classes
        self.dataset_simple = dataset_simple
        self.n_classes = len(dataset.classes)
        self.targets = np.array(dataset.targets)

        # cue information
        self.cue_proportion, self.randomize_cue, self.randomize_img = cue_proportion, randomize_cue, randomize_img
        self.cue_ids = get_cue_ids(targets=self.targets, n_classes=self.n_classes, prob=cue_proportion)

        # association IDs
        self.association_ids = get_dominoes_associations(targets_c10=self.targets, targets_fmnist=np.array(dataset_simple.targets))

    # dataset length
    def __len__(self):
        return len(self.dataset)

    # retrieve next sample
    def __getitem__(self, item):
        image, label = self.dataset[item]
        associated_id = self.association_ids[label][item]
        image = self.dataset[np.random.randint(0, len(self.dataset))][0] if self.randomize_img else image

        if self.cue_proportion > 0:
            put_cue_attribute = (np.random.uniform() < self.cue_proportion) if self.cue_ids is None else self.cue_ids[label][item]
        else:
            put_cue_attribute = False

        if put_cue_attribute:
            image_fmnist = self.dataset_simple[np.random.randint(0, len(self.dataset))][0] if self.randomize_cue else self.dataset_simple[associated_id][0]
        else:
            image_fmnist = torch.zeros_like(image)
        image = torch.cat((image_fmnist, image), dim=1)        

        return image, label


        
### Transforms for data 
class image3d(object):
    def __call__(self, img):
        img = img.convert('RGB')
        return img

def get_transform(tform_type='nocue'):

    if(tform_type == 'nocue'):
        train_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    elif(tform_type == 'dominoes'):
        train_transform = T.Compose([
            image3d(),
            T.Resize((32, 32)),
            T.ToTensor(),
        ])

    return train_transform


### Dataloaders
def get_dataloader(load_type='train', base_dataset='CIFAR10', cue_type='nocue', cue_proportion=1.0, randomize_cue=False, 
                    randomize_img=False, batch_size=128, data_dir='./datasets', subset_ids=None):

    if base_dataset == 'Dominoes':
        base_dataset = 'CIFAR10'
        cue_proportion = 0.0 if cue_type == 'nocue' else cue_proportion
        cue_type = 'dominoes'

    # define base dataset (pick train or test)
    dset_type = getattr(torchvision.datasets, base_dataset)
    dset = dset_type(root=f'{data_dir}/{base_dataset.lower()}/', 
                     train=(load_type == 'train'), download=True, transform=get_transform('nocue'))

    # pick cue
    if (cue_type == 'nocue'):
        pass
    elif (cue_type == 'box'):
        dset = boxDataset(dset, cue_proportion=cue_proportion, randomize_cue=randomize_cue, randomize_img=randomize_img)
    elif (cue_type == 'dominoes'):
        dset_type = getattr(torchvision.datasets, 'FashionMNIST')
        dset_simple = dset_type(root=f'{data_dir}/FashionMNIST/', 
                        train=(load_type == 'train'), download=True, transform=get_transform('dominoes'))

        dset = domDataset(dset, dset_simple, cue_proportion=cue_proportion, randomize_cue=randomize_cue, randomize_img=randomize_img)

    if isinstance(subset_ids, np.ndarray):
        dset = torch.utils.data.Subset(dset, subset_ids)

    # define dataloader
    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=(load_type=='train'), num_workers=4)
    return dataloader


#### Fix which samples of dataset will have cues
def get_cue_ids(targets=None, n_classes=10, prob=1.):
    cue_ids = {}
    for class_num in range(n_classes):
        idx = np.where(targets == class_num)[0]
        make_these_withcue = np.array([True]*idx.shape[0])
        make_these_withcue[int(idx.shape[0] * prob):] = False

        cue_ids.update({class_num: {idx[sample_id]: make_these_withcue[sample_id] for sample_id in range(idx.shape[0])}})

    return cue_ids


#### Dominoes data dictionaries
def get_dominoes_associations(targets_fmnist, targets_c10):
    association_ids = {i: 0 for i in range(10)}
    for class_num in range(10):
        idx_c10 = np.where(targets_c10 == class_num)[0]
        idx_fmnist = np.where(targets_fmnist == class_num)[0]
        association_ids[class_num] = {idx_c10[i]: idx_fmnist[i] for i in range(len(targets_c10) // 10)}
    return association_ids