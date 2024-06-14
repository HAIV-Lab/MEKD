from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import os
from resnet import *
from vgg import *
from mobilenet import *
from syntheticdigits import *
from svhn import *


class PartialDataset(Dataset):
    def __init__(self, ori_dset, rate):
        idx = np.random.choice(range(len(ori_dset)), replace=False, size=int(rate*len(ori_dset)))
        self.dataset = [ori_dset[i] for i in idx]
        self.data = [ori_dset.data[i] for i in idx]
        self.targets = [ori_dset.targets[i] for i in idx]
        self.labels = self.targets
        self.transform = ori_dset.transform
        self.target_transform = ori_dset.target_transform

    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)


def build_dataset(args):    
    if args.dataset == 'MNIST':
        if args.lamda == 1.0:
            train_loader =  DataLoader(
                    dataset = dset.MNIST(
                        root = './data/mnist/', train = True, download = True,
                        transform = transforms.Compose([
                            transforms.Resize(32),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])
                    ),
                    batch_size = args.batch_size,
                    shuffle = True
                )
        else:
            train_loader = DataLoader(
                dataset = PartialDataset(
                    ori_dset=dset.MNIST(
                        root = './data/mnist/', train = True, download = True,
                        transform = transforms.Compose([
                            transforms.Resize(32),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])
                    ),
                    rate = args.lamda
                ),
                batch_size = args.batch_size,
                shuffle=True
            )
        test_loader = DataLoader(
            dataset = dset.MNIST(
                root = './data/mnist/', train = False, download = True,
                transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
            ),
            batch_size = args.batch_size,
            shuffle = False
        )

    elif args.dataset == 'SynDigits':
        train_loader =  DataLoader(
            dataset = SyntheticDigits(
                root = './data/syntheticdigits/', 
                train = True, 
                download = True,
                transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
            ),
            batch_size = 64,
            shuffle = True
        )
        test_loader = DataLoader(
            dataset = SyntheticDigits(
                root = './data/syntheticdigits/', 
                train = False, 
                download = True,
                transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
            ),
            batch_size = 64,
            shuffle = False
        )

    elif args.dataset == 'SVHN':
        if args.lamda == 1.0:
            train_loader =  DataLoader(
                dataset = SVHN(
                    root = './data/svhn/', 
                    split = 'train', 
                    download = True,
                    transform = transforms.Compose([
                        transforms.Resize(32),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (0.5,))
                    ])
                ),
                batch_size = 64,
                shuffle = True
            ) 
        else:
            train_loader = DataLoader(
                dataset = PartialDataset(
                    ori_dset=SVHN(
                        root = './data/svhn/', 
                        split = 'train', 
                        download = True,
                        transform = transforms.Compose([
                            transforms.Resize(32),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,), (0.5,))
                        ])
                    ),
                    rate = args.lamda
                ),
                batch_size = args.batch_size,
                shuffle=True
            )
        test_loader = DataLoader(
            dataset = SVHN(
                root = './data/svhn/', 
                split = 'test', 
                download = True,
                transform = transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))
                ])
            ),
            batch_size = 64,
            shuffle = False
        )

    else:
        raise NotImplementedError
    
    return train_loader, test_loader


def build_model(args):
    if args.model == 'ResNet8':
        return resnet8(num_classes=args.num_classes, in_channels=args.channels)
    elif args.model == 'ResNet20':
        return resnet20(num_classes=args.num_classes, in_channels=args.channels)
    elif args.model == 'ResNet32':
        return resnet32(num_classes=args.num_classes, in_channels=args.channels)
    elif args.model == 'ResNet56':
        return resnet56(num_classes=args.num_classes, in_channels=args.channels)
    elif args.model == 'ResNet110':
        return resnet110(num_classes=args.num_classes, in_channels=args.channels)
    
    elif args.model == 'VGG19':
        return vgg19(num_classes=args.num_classes, in_channels=args.channels)
    elif args.model == 'VGG13':
        return vgg13(num_classes=args.num_classes, in_channels=args.channels)
    elif args.model == 'VGG11':
        return vgg11(num_classes=args.num_classes, in_channels=args.channels)
    elif args.model == 'VGG19_BN':
        return vgg19_bn(num_classes=args.num_classes, in_channels=args.channels)
    elif args.model == 'VGG13_BN':
        return vgg13_bn(num_classes=args.num_classes, in_channels=args.channels)
    elif args.model == 'VGG11_BN':
        return vgg11_bn(num_classes=args.num_classes, in_channels=args.channels)
    
    elif args.model == 'MobileNet':
        return mobilenet(num_classes=args.num_classes, in_channels=args.channels)

    else:
        raise NotImplementedError


def build_teacher(args):
    if args.teacher == 'ResNet8':
        teacher = resnet8(num_classes=args.num_classes, in_channels=args.channels)
    elif args.teacher == 'ResNet20':
        teacher = resnet20(num_classes=args.num_classes, in_channels=args.channels)
    elif args.teacher == 'ResNet32':
        teacher = resnet32(num_classes=args.num_classes, in_channels=args.channels)
    elif args.teacher == 'ResNet56':
        teacher = resnet56(num_classes=args.num_classes, in_channels=args.channels)
    elif args.teacher == 'ResNet110':
        teacher = resnet110(num_classes=args.num_classes, in_channels=args.channels)
    
    elif args.teacher == 'VGG19':
        teacher = vgg19(num_classes=args.num_classes, in_channels=args.channels)
    elif args.teacher == 'VGG13':
        teacher = vgg13(num_classes=args.num_classes, in_channels=args.channels)
    elif args.teacher == 'VGG11':
        teacher = vgg11(num_classes=args.num_classes, in_channels=args.channels)
    elif args.teacher == 'VGG19_BN':
        teacher = vgg19_bn(num_classes=args.num_classes, in_channels=args.channels)
    elif args.teacher == 'VGG13_BN':
        teacher = vgg13_bn(num_classes=args.num_classes, in_channels=args.channels)
    elif args.teacher == 'VGG11_BN':
        teacher = vgg11_bn(num_classes=args.num_classes, in_channels=args.channels)
    
    elif args.teacher == 'MobileNet':
        teacher = mobilenet(num_classes=args.num_classes, in_channels=args.channels)
    
    else:
        raise NotImplementedError

    if args.dataset == 'SVHN':
        teacher_path = os.path.join('./output/teacher/', 'SynDigits', args.teacher, 'teacher.pth')
    else:
        teacher_path = os.path.join('./output/teacher/', args.dataset, args.teacher, 'teacher.pth')
    teacher.load_state_dict(torch.load(teacher_path)['model'])
    return teacher
