import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import pandas as pd
import os
import nibabel


class ECG1D(torch.utils.data.Dataset):
    def __init__(self, directory, test=False):
        super().__init__()

        if not test:
            self.df = pd.read_csv(directory + '/mitbih_train.csv')
        else:
            self.df = pd.read_csv(directory + '/mitbih_test.csv')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx, :-1].values.astype(float)
        label = self.df.iloc[idx, -1]
        sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(dim=0)
        label = torch.tensor(label, dtype=torch.long)

        return sample, label


class BRATSVolumes(torch.utils.data.Dataset):
    def __init__(self, directory,  normalize=None, img_size=32):
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.normalize = normalize or (lambda x: x)
        self.img_size = img_size
        self.seqtypes = ['t1n', 't1c', 't2w', 't2f', 'seg']
        self.seqtypes_set = set(self.seqtypes)
        self.database = []

        for root, dirs, files in os.walk(self.directory):
            # Ensure determinism
            dirs.sort()
            files.sort()
            # if there are no subdirs, we have a datadir
            if not dirs:
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('-')[4].split('.')[0]
                    datapoint[seqtype] = os.path.join(root, f)
                self.database.append(datapoint)

    def __getitem__(self, x):
        filedict = self.database[x]
        name = filedict['t1n']
        nib_img = nibabel.load(name)  # We only use t1 weighted images
        out = nib_img.get_fdata()

        # Clip and normalize the images
        out_clipped = np.clip(out, np.quantile(out, 0.001), np.quantile(out, 0.999))
        out_normalized = (out_clipped - np.min(out_clipped)) / (np.max(out_clipped) - np.min(out_clipped))
        out = torch.tensor(out_normalized)

        # Zero pad images
        image = torch.zeros(1, 256, 256, 256)
        image[:, 8:-8, 8:-8, 50:-51] = out

        # Downsampling
        if self.img_size == 32:
            downsample = nn.AvgPool3d(kernel_size=8, stride=8)
            image = downsample(image)

        # Normalization
        image = self.normalize(image)

        # Insert dummy label
        label = 1

        return image, label

    def __len__(self):
        return len(self.database)


class LIDCVolumes(torch.utils.data.Dataset):
    def __init__(self, directory, normalize=None, img_size=32):
        super().__init__()
        self.directory = os.path.expanduser(directory)
        self.normalize = normalize or (lambda x: x)
        self.img_size = img_size
        self.database = []

        for root, dirs, files in os.walk(self.directory):
            # Ensure determinism
            dirs.sort()
            files.sort()
            # if there are no subdirs, we have a datadir
            if not dirs:
                datapoint = dict()
                for f in files:
                    datapoint['image'] = os.path.join(root, f)
                if len(datapoint) != 0:
                    self.database.append(datapoint)

    def __getitem__(self, x):
        filedict = self.database[x]
        name = filedict['image']
        nib_img = nibabel.load(name)
        out = nib_img.get_fdata()

        out = torch.Tensor(out)

        image = torch.zeros(1, 256, 256, 256)
        image[:, :, :, :] = out

        if self.img_size == 32:
            downsample = nn.AvgPool3d(kernel_size=8, stride=8)
            image = downsample(image)

        # normalization
        image = self.normalize(image)

        # Insert dummy label
        label = 1

        return image, label

    def __len__(self):
        return len(self.database)


def get_dataset(args, only_test=False, all=False):
    train_set = None
    val_set = None
    test_set = None

    if args.dataset == 'chestmnist':
        from medmnist import ChestMNIST
        transforms = T.Compose([
            T.ToTensor(),
            T.Grayscale()
        ])

        train_set = ChestMNIST(split='train', transform=transforms, download='True', size=args.img_size)
        val_set = ChestMNIST(split='val', transform=transforms, download='True', size=args.img_size)
        test_set = ChestMNIST(split='test', transform=transforms, download='True', size=args.img_size)

        print(f'Training set containing {len(train_set)} images.')
        print(f'Test set containing {len(test_set)} images.')

        args.data_type = 'img'
        args.in_size, args.out_size = 2, 1
        args.data_size = (1, args.img_size, args.img_size)

    elif args.dataset == 'pneumoniamnist':
        from medmnist import PneumoniaMNIST
        transforms = T.Compose([
            T.ToTensor(),
            T.Grayscale()
        ])

        train_set = PneumoniaMNIST(split='train', transform=transforms, download='True', size=args.img_size)
        val_set = PneumoniaMNIST(split='val', transform=transforms, download='True', size=args.img_size)
        test_set = PneumoniaMNIST(split='test', transform=transforms, download='True', size=args.img_size)

        print(f'Training set containing {len(train_set)} images.')
        print(f'Test set containing {len(test_set)} images.')

        args.data_type = 'img'
        args.in_size, args.out_size = 2, 1
        args.data_size = (1, args.img_size, args.img_size)

    elif args.dataset == 'retinamnist':
        from medmnist import RetinaMNIST
        transforms = T.Compose([
            T.ToTensor(),
        ])
        train_set = RetinaMNIST(split='train', transform=transforms, download='True', size=args.img_size)
        val_set = RetinaMNIST(split='val', transform=transforms, download='True', size=args.img_size)
        test_set = RetinaMNIST(split='test', transform=transforms, download='True', size=args.img_size)

        print(f'Training set containing {len(train_set)} images.')
        print(f'Test set containing {len(test_set)} images.')

        args.data_type = 'img'
        args.in_size, args.out_size = 2, 3
        args.data_size = (3, args.img_size, args.img_size)

    elif args.dataset == 'dermamnist':
        from medmnist import DermaMNIST
        transforms = T.Compose([
            T.ToTensor(),
        ])
        train_set = DermaMNIST(split='train', transform=transforms, download='True', size=args.img_size)
        val_set = DermaMNIST(split='val', transform=transforms, download='True', size=args.img_size)
        test_set = DermaMNIST(split='test', transform=transforms, download='True', size=args.img_size)

        print(f'Training set containing {len(train_set)} images.')
        print(f'Test set containing {len(test_set)} images.')

        args.data_type = 'img'
        args.in_size, args.out_size = 2, 3
        args.data_size = (3, args.img_size, args.img_size)

    elif args.dataset == 'octmnist':
        from medmnist import OCTMNIST
        transforms = T.Compose([
            T.ToTensor(),
            T.Grayscale()
        ])
        train_set = OCTMNIST(split='train', transform=transforms, download='True', size=args.img_size)
        val_set = OCTMNIST(split='val', transform=transforms, download='True', size=args.img_size)
        test_set = OCTMNIST(split='test', transform=transforms, download='True', size=args.img_size)

        print(f'Training set containing {len(train_set)} images.')
        print(f'Test set containing {len(test_set)} images.')

        args.data_type = 'img'
        args.in_size, args.out_size = 2, 1
        args.data_size = (1, args.img_size, args.img_size)

    elif args.dataset == 'pathmnist':
        from medmnist import PathMNIST
        transforms = T.Compose([
            T.ToTensor(),
        ])
        train_set = PathMNIST(split='train', transform=transforms, download='True', size=args.img_size)
        val_set = PathMNIST(split='val', transform=transforms, download='True', size=args.img_size)
        test_set = PathMNIST(split='test', transform=transforms, download='True', size=args.img_size)

        print(f'Training set containing {len(train_set)} images.')
        print(f'Test set containing {len(test_set)} images.')

        args.data_type = 'img'
        args.in_size, args.out_size = 2, 3
        args.data_size = (3, args.img_size, args.img_size)

    elif args.dataset == 'tissuemnist':
        from medmnist import TissueMNIST
        transforms = T.Compose([
            T.ToTensor(),
            T.Grayscale()
        ])
        train_set = TissueMNIST(split='train', transform=transforms, download='True', size=args.img_size)
        val_set = TissueMNIST(split='val', transform=transforms, download='True', size=args.img_size)
        test_set = TissueMNIST(split='test', transform=transforms, download='True', size=args.img_size)

        print(f'Training set containing {len(train_set)} images.')
        print(f'Test set containing {len(test_set)} images.')

        args.data_type = 'img'
        args.in_size, args.out_size = 2, 1
        args.data_size = (1, args.img_size, args.img_size)

    elif args.dataset == 'nodulemnist':
        from medmnist import NoduleMNIST3D
        train_set = NoduleMNIST3D(split='train', download='True', size=args.img_size)
        val_set = NoduleMNIST3D(split='val', download='True', size=args.img_size)
        test_set = NoduleMNIST3D(split='test', download='True', size=args.img_size)

        print(f'Training set containing {len(train_set)} images.')
        print(f'Test set containing {len(test_set)} images.')

        args.data_type = 'img3d'
        args.in_size, args.out_size = 3, 1
        args.data_size = (1, args.img_size, args.img_size, args.img_size)

    elif args.dataset == 'organmnist':
        from medmnist import OrganMNIST3D
        train_set = OrganMNIST3D(split='train', download='True', size=args.img_size)
        val_set = OrganMNIST3D(split='val', download='True', size=args.img_size)
        test_set = OrganMNIST3D(split='test', download='True', size=args.img_size)

        print(f'Training set containing {len(train_set)} images.')
        print(f'Test set containing {len(test_set)} images.')

        args.data_type = 'img3d'
        args.in_size, args.out_size = 3, 1
        args.data_size = (1, args.img_size, args.img_size, args.img_size)

    elif args.dataset == 'vesselmnist':
        from medmnist import VesselMNIST3D
        train_set = VesselMNIST3D(split='train', download='True', size=args.img_size)
        val_set = VesselMNIST3D(split='val', download='True', size=args.img_size)
        test_set = VesselMNIST3D(split='test', download='True', size=args.img_size)

        print(f'Training set containing {len(train_set)} images.')
        print(f'Test set containing {len(test_set)} images.')

        args.data_type = 'img3d'
        args.in_size, args.out_size = 3, 1
        args.data_size = (1, args.img_size, args.img_size, args.img_size)

    elif args.dataset == 'brats':
        dataset = BRATSVolumes('/path/to/brats/dataset', img_size=args.img_size)

        # Define split sizes
        train_size = int(0.8 * len(dataset))  # 80% for training
        test_size = len(dataset) - train_size  # 20% for testing

        generator = torch.Generator().manual_seed(42)
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)

        print(f'Training set containing {len(train_set)} images.')
        print(f'Test set containing {len(test_set)} images.')

        args.data_type = 'img3d'
        args.in_size, args.out_size = 3, 1
        args.data_size = (1, args.img_size, args.img_size, args.img_size)

    elif args.dataset == 'lidc-idri':
        dataset = LIDCVolumes('/path/to/lidc-idri/dataset', img_size=args.img_size)

        # Define split sizes
        train_size = int(0.8 * len(dataset))  # 80% for training
        test_size = len(dataset) - train_size  # 20% for testing

        generator = torch.Generator().manual_seed(42)
        train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)

        print(f'Training set containing {len(train_set)} images.')
        print(f'Test set containing {len(test_set)} images.')

        args.data_type = 'img3d'
        args.in_size, args.out_size = 3, 1
        args.data_size = (1, args.img_size, args.img_size, args.img_size)

    elif args.dataset == 'ecg':
        train_set = ECG1D('/path/to/ecg/dataset', test=False)
        val_set = ECG1D('/path/to/ecg/dataset', test=True)  # DUMMY VALIDATION SET (DELETE AFTER CREATION)
        test_set = ECG1D('/path/to/ecg/dataset', test=True)

        print(f'Training set containing {len(train_set)} ECG signals.')
        print(f'Test set containing {len(test_set)} ECG signals.')

        args.data_type = 'timeseries'
        args.in_size, args.out_size = 1, 1
        args.data_size = (1, args.img_size)

    else:
        raise NotImplementedError()

    if only_test:
        return test_set

    elif all:
        return train_set, val_set, test_set

    else:
        return train_set, test_set
