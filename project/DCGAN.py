import os
import time
import torch
from tools import Logger
from build import build_dataset, build_teacher
import argparse
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchvision.utils import save_image


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset type')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

parser.add_argument('--model', type=str, default='DCGAN', help='Model type')
parser.add_argument('--teacher', type=str, default='ResNet32', help='Teacher type')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
parser.add_argument('--img_size', type=int, default=32, help='Image size')
parser.add_argument('--latent_dim', type=int, default=10, help='Latent dimension')
parser.add_argument('--channels', type=int, default=1, help='Channels of image')

parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--w_im', type=float, default=0.5, help='Weight of IM loss')
parser.add_argument('--epochs', type=int, default=200, help='Epochs')
parser.add_argument('--lamda', type=float, default=0.1, help='Ratio')

parser.add_argument('--interval', type=int, default=100, help='Interval')
parser.add_argument('--save_model', type=str, default='./output/MEKD/', help='Path to save model')
args = parser.parse_args()
args.save_model = os.path.join(args.save_model, args.dataset, args.model)


logger = Logger(os.path.join(args.save_model, '{}.log'.format(time.strftime('%Y%m%d%H%M%S', time.localtime()))))
logger.info(args)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_loader, test_loader = build_dataset(args)
teacher = build_teacher(args).to(device)
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


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(args.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = args.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


generator = Generator()
discriminator = Discriminator()
generator.to(device)
discriminator.to(device)
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
adversarial_loss = torch.nn.BCELoss().to(device)


Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
for epoch in range(args.epochs):
    for i, (imgs, _) in enumerate(train_loader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        fake_loss = adversarial_loss(discriminator(gen_imgs), fake)

        d_loss = real_loss + fake_loss

        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_D.zero_grad()
        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_adv_loss = adversarial_loss(discriminator(gen_imgs), valid)

        # Loss measures generated images quality related to teacher model
        with torch.no_grad():
            output = teacher(gen_imgs)['output']
        output = torch.max(output, dim=-1)[1]
        g_im_loss = torch.mean(- output * torch.log(discriminator(gen_imgs) + 1e-5))

        g_loss = g_adv_loss + g_im_loss * args.w_im

        g_loss.backward()
        optimizer_G.step()

        if i % args.interval == 0:
            logger.info(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, args.epochs, i, len(train_loader), d_loss.item(), g_loss.item())
            )
    
    if args.save_model:
        torch.save(
            {
                'model': generator.state_dict(),
                'info': {
                    'epoch': epoch
                }
            },
            os.path.join(args.save_model, 'G_{}.pth'.format(epoch))
        )

        os.makedirs(os.path.join(args.save_model, 'imgs'), exist_ok=True)
        save_image(
            gen_imgs.data[:25], 
            os.path.join(args.save_model, 'imgs', '{}.png'.format(epoch)), 
            nrow=5, normalize=True
        )
