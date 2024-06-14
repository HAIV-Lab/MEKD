import os
import time
import torch
from tools import Logger
from build import build_model, build_dataset, build_teacher
import argparse
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset type')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')

parser.add_argument('--teacher', type=str, default='ResNet32', help='Teacher type')
parser.add_argument('--model', type=str, default='ResNet8', help='Student type')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
parser.add_argument('--channels', type=int, default=1, help='Channels of image')

parser.add_argument('--lr_G', type=float, default=0.2, help='Learning rate')
parser.add_argument('--lr_S', type=float, default=2e-3, help='Learning rate')

parser.add_argument('--epochs', type=int, default=240, help='Epochs')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--lamda', type=float, default=0.1, help='Ratio')

parser.add_argument('--oh', type=float, default=1, help='one hot loss')
parser.add_argument('--ie', type=float, default=5, help='information entropy loss')
parser.add_argument('--a', type=float, default=0.1, help='activation loss')

parser.add_argument('--save_model', type=str, default='./output/DAFL/', help='Path to save model')
args = parser.parse_args()
args.save_model = os.path.join(args.save_model, args.dataset, '{}_to_{}'.format(args.teacher, args.model))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = args.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(args.latent_dim, 128*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(128),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, args.channels, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(args.channels, affine=False) 
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img


logger = Logger(os.path.join(args.save_model, '{}.log'.format(time.strftime('%Y%m%d%H%M%S', time.localtime()))))
logger.info(args)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_loader, test_loader = build_dataset(args)
model = build_model(args).to(device)
teacher = build_teacher(args).to(device)
generator = Generator().to(device)
if args.dataset == 'MNIST':
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_G)
    optimizer_S = torch.optim.Adam(model.parameters(), lr=args.lr_S)
else:
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_G)
    optimizer_S = torch.optim.SGD(model.parameters(), lr=args.lr_S, momentum=0.9, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss().to(device)

def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, size_average=False)  / y.shape[0]
    return l_kl

def adjust_learning_rate(optimizer, epoch, learing_rate):
    if epoch < 800:
        lr = learing_rate
    elif epoch < 1600:
        lr = 0.1*learing_rate
    else:
        lr = 0.01*learing_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


max_acc = 0
for epoch in range(args.epochs):
    model.train()
    teacher.eval()
    generator.train()

    if args.dataset != 'MNIST':
        adjust_learning_rate(optimizer_S, epoch, args.lr_S)

    for i in range(120):
        z = Variable(torch.randn(args.batch_size, args.latent_dim)).to(device)
        optimizer_G.zero_grad()
        optimizer_S.zero_grad()
        gen_imgs = generator(z)
        response = teacher(gen_imgs)
        outputs_T, features_T = response['output'], response['feature']
        pred = outputs_T.data.max(1)[1]
        loss_activation = -features_T.abs().mean()
        loss_one_hot = criterion(outputs_T,pred)
        softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
        loss_information_entropy = (softmax_o_T * torch.log10(softmax_o_T)).sum()
        loss = loss_one_hot * args.oh + loss_information_entropy * args.ie + loss_activation * args.a
        loss_kd = kdloss(model(gen_imgs.detach())['output'], outputs_T.detach())
        loss += loss_kd       
        loss.backward()
        optimizer_G.step()
        optimizer_S.step()
        if i == 1:
            logger.info("[Epoch %d/%d] [loss_oh: %f] [loss_ie: %f] [loss_a: %f] [loss_kd: %f]" % (epoch, args.epochs,loss_one_hot.item(), loss_information_entropy.item(), loss_activation.item(), loss_kd.item()))


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
