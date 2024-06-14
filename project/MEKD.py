import os
import time
import torch
from tools import Logger
from build import build_model, build_dataset, build_teacher
import argparse
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset type')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')

parser.add_argument('--teacher', type=str, default='ResNet32', help='Teacher type')
parser.add_argument('--model', type=str, default='ResNet8', help='Student type')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
parser.add_argument('--img_size', type=int, default=32, help='Image size')
parser.add_argument('--channels', type=int, default=1, help='Channels of image')

parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Wright decay')

parser.add_argument('--epochs', type=int, default=240, help='Epochs')
parser.add_argument('--milestones', type=int, nargs='+', default=[80, 160], help='Milestones')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma')
parser.add_argument('--lamda', type=float, default=0.1, help='Ratio')

parser.add_argument('--res_type', type=str, default='soft', choices=['soft', 'hard'], help='Responses type')
parser.add_argument('--t', type=float, default=5.0, help='Temperature')
parser.add_argument('--gen_t', type=float, default=5.0, help='Temperature of generation')
parser.add_argument('--w_f', type=float, default=1.0, help='Weight of loss F')
parser.add_argument('--w_kl', type=float, default=1.0, help='Weight of loss KLDiv')
parser.add_argument('--F', type=int, default=1, choices=[1,2], help='L1 or L2 loss')

parser.add_argument('--interval', type=int, default=100, help='Interval')
parser.add_argument('--save_model', type=str, default='./output/MEKD/', help='Path to save model')
parser.add_argument('--G_path', type=str, default='DCGAN/G_175.pth', help='Generator path')
args = parser.parse_args()
args.G_model = os.path.join('./output/MEKD/', args.dataset, args.G_path)
args.save_model = os.path.join(args.save_model, args.dataset, '{}_to_{}'.format(args.teacher, args.model))
args.latent_dim = args.num_classes


logger = Logger(os.path.join(args.save_model, '{}_{}.log'.format(time.strftime('%Y%m%d%H%M%S', time.localtime()), np.random.randint(0, 100))))
logger.info(args)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
_, test_loader = build_dataset(args)
model = build_model(args).to(device)
teacher = build_teacher(args).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
KLDivLoss = torch.nn.KLDivLoss(reduction='batchmean').to(device)
if args.F == 1:
    LFloss = torch.nn.L1Loss().to(device)
else:
    LFloss = torch.nn.MSELoss().to(device)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = args.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(args.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, args.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

generator = Generator()
generator.load_state_dict(torch.load(args.G_model)['model'])
generator.to(device)
generator.eval()


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


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
max_acc = 0
for epoch in range(args.epochs):
    model.train()
    teacher.eval()
    for i in range(500):
        z = Variable(Tensor(np.random.normal(0, 1, (args.batch_size, args.latent_dim))))
        data = generator(z).type(torch.FloatTensor).to(device)
        
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_output = teacher(data)['output']
            if args.res_type == 'hard':
                hard_output = torch.zeros(teacher_output.shape).to(device)
                hard_output[torch.arange(0, data.shape[0]), torch.max(teacher_output, dim=-1)[-1]] = 1.0
                teacher_output = hard_output
            teacher_generation = generator(teacher_output / args.gen_t)

        student_output = model(data)['output']
        student_generation = generator(student_output / args.gen_t)

        loss_LF = LFloss(student_generation, teacher_generation)
        loss_KL = KLDivLoss(
            torch.log(F.softmax(student_output / args.t, dim=-1)),
            F.softmax(teacher_output / args.t, dim=-1)
        ) * args.t
        loss = loss_LF * args.w_f + loss_KL * args.w_kl

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
                epoch, i, 500, 100.*i/500, loss.item())
            )
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
