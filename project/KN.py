import os
import time
import torch
from tools import Logger
from build import build_model, build_dataset, build_teacher
import argparse
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import ImageFolder, IMG_EXTENSIONS, default_loader
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset type')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

parser.add_argument('--teacher', type=str, default='ResNet32', help='Teacher type')
parser.add_argument('--model', type=str, default='ResNet8', help='Student type')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
parser.add_argument('--channels', type=int, default=1, help='Channels of image')

parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Wright decay')

parser.add_argument('--budget', type=int, default=0.2, help='Budget of transfer set')
parser.add_argument('--t', type=float, default=5.0, help='Temperature')

parser.add_argument('--epochs', type=int, default=240, help='Epochs')
parser.add_argument('--milestones', type=int, nargs='+', default=[80, 160], help='Milestones')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma')
parser.add_argument('--lamda', type=float, default=0.1, help='Ratio')

parser.add_argument('--interval', type=int, default=100, help='Interval')
parser.add_argument('--save_model', type=str, default='./output/KN/', help='Path to save model')
args = parser.parse_args()
args.save_model = os.path.join(args.save_model, args.dataset, '{}_to_{}'.format(args.teacher, args.model))


logger = Logger(os.path.join(args.save_model, '{}.log'.format(time.strftime('%Y%m%d%H%M%S', time.localtime()))))
logger.info(args)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_loader, test_loader = build_dataset(args)
model = build_model(args).to(device)
teacher = build_teacher(args).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
KLDivLoss = torch.nn.KLDivLoss(reduction='batchmean').to(device)


def get_transferset(model, dataset, batchsize, budget):
    transferset = []
    idx_set = set(range(len(dataset)))

    for _ in range(0, budget, batchsize):
        size = min(batchsize, budget - len(transferset))
        if len(idx_set) < size:
            idx_set = set(range(len(dataset)))

        idxs = np.random.choice(list(idx_set), replace=False, size=size)
        idx_set = idx_set - set(idxs)

        x_t = torch.stack([dataset[i][0] for i in idxs]).to(device)
        with torch.no_grad():
            y_t = model(x_t)['output'].cpu()

        if hasattr(dataset, 'samples'):
            img_t = [dataset.samples[i][0] for i in idxs]
        else:
            img_t = [dataset.data[i] for i in idxs]
            if isinstance(dataset.data[0], torch.Tensor):
                img_t = [x.numpy() for x in img_t]
        
        for i in range(x_t.size(0)):
            img_t_i = img_t[i].squeeze() if isinstance(img_t[i], np.ndarray) else img_t[i]
            transferset.append((img_t_i, y_t[i].cpu().numpy().squeeze()))
        
    return transferset


class TransferSetImagePaths(ImageFolder):
    """TransferSet Dataset, for when images are stored as *paths*"""

    def __init__(self, samples, transform=None, target_transform=None):
        self.loader = default_loader
        self.extensions = IMG_EXTENSIONS
        self.samples = samples
        self.targets = [s[1] for s in samples]
        self.transform = transform
        self.target_transform = target_transform


class TransferSetImages(Dataset):
    def __init__(self, samples, transform=None, target_transform=None):
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform

        self.data = [self.samples[i][0] for i in range(len(self.samples))]
        self.targets = [self.samples[i][1] for i in range(len(self.samples))]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = img.transpose(1,2,0)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


def transferset_to_dataset(samples, budget=None, transform=None, target_transform=None):
    # Images are either stored as paths, or numpy arrays
    sample_x = samples[0][0]
    assert budget <= len(samples), 'Required {} samples > Found {} samples'.format(budget, len(samples))

    if isinstance(sample_x, str):
        return TransferSetImagePaths(samples[:budget], transform=transform, target_transform=target_transform)
    elif isinstance(sample_x, np.ndarray):
        return TransferSetImages(samples[:budget], transform=transform, target_transform=target_transform)
    else:
        raise ValueError('type(x_i) ({}) not recognized. Supported types = (str, np.ndarray)'.format(type(sample_x)))


teacher.eval()
correct = 0
with torch.no_grad():
    for data, labels in test_loader:
        data = data.type(torch.FloatTensor).to(device)
        labels = labels.to(device)

        output = teacher(data)["output"]
        pred = output.argmax(dim=-1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()

logger.info('Test Accuracy of Teacher: {}/{} ({:.2f}%)'.format(
    correct, len(test_loader.dataset), 100.*correct/len(test_loader.dataset)
))


args.budget = int(args.budget * len(train_loader.dataset))
transferset = get_transferset(teacher, train_loader.dataset, args.batch_size, args.budget)
transfer_dataset = transferset_to_dataset(transferset, args.budget, transform=train_loader.dataset.transform, target_transform=train_loader.dataset.target_transform)
transfer_loader = DataLoader(transfer_dataset, batch_size=args.batch_size, shuffle=True)


max_acc = 0
t = args.t
for epoch in range(args.epochs):
    model.train()
    for i, (data, labels) in enumerate(transfer_loader):
        data = data.type(torch.FloatTensor).to(device)
        labels = labels.type(torch.FloatTensor).to(device)

        optimizer.zero_grad()
        output = model(data)["output"]
        loss = KLDivLoss(
            torch.log(F.softmax(output / t, dim=-1)),
            F.softmax(labels / t, dim=-1)
        ) * t
        loss.backward()
        optimizer.step()

        if i % args.interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tCE_loss: {:.6f}'.format(
                epoch, i*len(data), len(transfer_loader.dataset), 100.*i/len(transfer_loader), loss.item()
            ))
    scheduler.step()


    model.eval()
    correct = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.type(torch.FloatTensor).to(device)
            labels = labels.to(device)

            output = model(data)["output"]
            pred = output.argmax(dim=-1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
    
    logger.info('Test Accuracy: {}/{} ({:.2f}%)'.format(
        correct, len(test_loader.dataset), 100.*correct/len(test_loader.dataset)
    ))

    
    if 100.*correct/len(test_loader.dataset) > max_acc:
        max_acc = 100.*correct/len(test_loader.dataset)
        if args.save_model:
            if not os.path.exists(args.save_model):
                os.makedirs(args.save_model)
            torch.save(
                {
                    'model': model.state_dict(),
                    'info': {
                        'epoch': epoch,
                        'test_accuracy': 100.*correct/len(test_loader.dataset),
                    },
                },
                os.path.join(args.save_model, 'student.pth')
            )


logger.info('Max Test Accuracy : {}'.format(max_acc))
