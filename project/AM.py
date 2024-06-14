import os
import time
from PIL import Image
import torch
from tools import Logger
from build import build_model, build_dataset, build_teacher
import argparse
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset type')
parser.add_argument('--batch_size', type=int, default=1000, help='Batch size')

parser.add_argument('--teacher', type=str, default='ResNet32', help='Teacher type')
parser.add_argument('--model', type=str, default='ResNet8', help='Student type')
parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
parser.add_argument('--channels', type=int, default=1, help='Channels of image')

parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--real_images', type=int, default=5000, help='Number of unlabeled real images')
parser.add_argument('--T', type=int, default=3, help='Circle times')
parser.add_argument('--lamda', type=float, default=0.1, help='Ratio')

parser.add_argument('--interval', type=int, default=100, help='Interval')
parser.add_argument('--save_model', type=str, default='./output/AM/', help='Path to save model')
args = parser.parse_args()
args.save_model = os.path.join(args.save_model, args.dataset, '{}_to_{}'.format(args.teacher, args.model))
args.epochs = [80, 60, 40, 20]
args.activate_interval = int(5e5 / args.batch_size / 5)


logger = Logger(os.path.join(args.save_model, '{}.log'.format(time.strftime('%Y%m%d%H%M%S', time.localtime()))))
logger.info(args)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_loader, test_loader = build_dataset(args)
train_dataset = train_loader.dataset
model = build_model(args).to(device)
teacher = build_teacher(args).to(device)
max_acc = 0

def cross_entropy(pred, soft_targets):
    logsoftmax = torch.nn.LogSoftmax(dim=-1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))


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


class ArrayDataset(Dataset):
    def __init__(self, samples, labels=None, logits=None):
        self.samples = samples
        self.labels = labels
        self.logits = logits

    def __getitem__(self, index):
        img = self.samples[index]
        img = torch.from_numpy(img)

        if self.labels is not None:
            label = self.labels[index]
        else:
            label = 0
        
        if self.logits is not None:
            logit = self.logits[index]
        else:
            logit = 0

        return img, logit, label
 
    def __len__(self):
        return len(self.samples)


class PairDataset(Dataset):
    def __init__(self, ori_dataset, samples_index_1, samples_index_2):
        self.ori_dataset = ori_dataset
        self.samples_index_1 = samples_index_1
        self.samples_index_2 = samples_index_2

    def __getitem__(self, index):
        img_index_1 = self.samples_index_1[index]
        img1 = self.ori_dataset[img_index_1][0]

        img_index_2 = self.samples_index_2[index]
        img2 = self.ori_dataset[img_index_2][0]

        return img1, img2, img_index_1, img_index_2

    def __len__(self):
        return len(self.samples_index_1)


def train():
    best_acc = 0
    img_train_loader = DataLoader(
            ArrayDataset(x_labeled_array, y_labeled_label, y_labeled_logits),
            batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5.0e-4)
    c = 1
    for epoch in range(sum(args.epochs)):
        model.train()
        if epoch == sum(args.epochs[:c]):
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 10.
            c += 1
        for i, (data, logits, targets) in enumerate(img_train_loader):
            data = data.type(torch.FloatTensor).to(device)
            logits = logits.type(torch.FloatTensor).to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            output = model(data)["output"]
            loss = cross_entropy(output, logits)
            loss.backward()
            optimizer.step()

            if i % args.interval == 0:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] \t CE_loss: {:.6f}'.format(
                    epoch, i*len(data), len(img_train_loader.dataset), 100.*i/len(img_train_loader), loss.item()
                ))

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

        if epoch > 80 and 100.*correct/len(test_loader.dataset) > best_acc:
            best_acc = 100.*correct/len(test_loader.dataset)
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
        
        global max_acc
        if epoch > 80 and 100.*correct/len(test_loader.dataset) > max_acc:
            max_acc = 100.*correct/len(test_loader.dataset)
    
    logger.info('Best Test Accuracy : {}'.format(best_acc))


################ select partial dataset of number 'args.real_images' ######################
x_oris = np.random.choice(range(len(train_dataset)), replace=False, size=args.real_images)
x_oris_img = np.stack([train_dataset[i][0].data.cpu().numpy() for i in x_oris])
x_oris_1 = []
x_oris_2 = []
for shift in range(int((args.real_images - 1) / 2.)):
    for index in range(x_oris.shape[0]):
        x_oris_shift = np.roll(x_oris, shift+1)
        x_oris_1.append(x_oris[index])
        x_oris_2.append(x_oris_shift[index])
logger.info('Number of real images: {} \t Number of images pair: {} and {}'.format(
    len(x_oris_img), len(x_oris_1), len(x_oris_2)
))


################ get labels and logits of real images ######################
x_labeled_array = []
y_labeled_label = []
y_labeled_logits = []
val_loader = DataLoader(ArrayDataset(x_oris_img), batch_size=args.batch_size, shuffle=False)

with torch.no_grad():
    for img, _, _ in val_loader:
        img = img.type(torch.FloatTensor).to(device)
        logits = teacher(img)['output']
        probs = F.softmax(logits, dim=-1)
        target = torch.max(probs, dim=-1)[1]

        x_labeled_array.append(img.data.cpu().numpy())
        y_labeled_label.append(target.data.cpu().numpy())
        y_labeled_logits.append(probs.data.cpu().numpy())

x_labeled_array = np.concatenate(x_labeled_array, axis=0)
y_labeled_label = np.concatenate(y_labeled_label, axis=0)
y_labeled_logits = np.concatenate(y_labeled_logits, axis=0)

logger.info('Number of real images: {} \t Number of labels: {} \t Number of logits: {}'.format(
    len(x_labeled_array), len(y_labeled_label), len(y_labeled_logits)
))


################ train student model using real images ######################
train()


for t in range(args.T):
    ################ load student model ######################
    model.load_state_dict(torch.load(os.path.join(args.save_model, 'student.pth'))['model'])
    model.to(device)
    model.eval()


    ################ select mixup images using activaet learning ######################
    x_oris_1_array = []
    x_oris_2_array = []
    mix_image = []
    mix_images_weights = []
    unc_tags = []
    w = np.arange(0.30, 0.70+0.04, 0.04)
    mix_img_val_loader = DataLoader(PairDataset(train_dataset, x_oris_1, x_oris_2), batch_size=args.batch_size, shuffle=True)

    for batch_idx, (img_1, img_2, img_index_1, img_index_2) in enumerate(mix_img_val_loader):
        img_1 = img_1.type(torch.FloatTensor).to(device)
        img_2 = img_2.type(torch.FloatTensor).to(device)
        img_index_1 = img_index_1.data.cpu().numpy()
        img_index_2 = img_index_2.data.cpu().numpy()

        uncertainty = []
        best_w = []

        # x_oris_1_array.append(img_1.data.cpu().numpy())
        # x_oris_2_array.append(img_2.data.cpu().numpy())
        x_oris_1_array.append(img_index_1)
        x_oris_2_array.append(img_index_2)

        for i in range(len(w)):
            mix = img_1 * w[i] + img_2 * (1 - w[i]) # image merge; convex combination
            with torch.no_grad():
                logits = model(mix)['output'] # query teacher network to obtain logits
            probs = F.softmax(logits, dim=-1) # get probabilities with softmax
            predict, target = torch.max(probs, dim=1) # get predicted probability(predict) and class label(tag)
            uncertainty.append(predict.data.cpu().numpy()) # append probabilities to list for uncertainty sorting

        uncertainty = np.array(uncertainty) # uncertainty list to array
        indices = np.argmin(uncertainty, axis=0) # uncertainty sorting and obtain smallest probabilities as largest uncertainties
        uncertainty = uncertainty[[indices], np.arange(args.batch_size)].squeeze() # obtain corresponding sorted uncertainty result for each data; 100 is batch size.

        best_w = indices / (len(w) - 1) * (w[-1] - w[0]) + w[0] # find the correponding weights; here, indices = i in the for loop, so it could be used for correct weight recovery

        mix_images_weights.append(best_w) # element-wise multiplication for image
        unc_tags.append(uncertainty) # get corresponding uncertainty (probabilites)

        if batch_idx % (args.activate_interval) == 0:
            logger.info('Circle time: {} \t Activate Learning of {} / {} ({:.2f}%)'.format(
                t, batch_idx*len(img_1), 5e5, 100.*batch_idx*len(img_1)/5e5
            ))
        
        if (batch_idx + 1) * args.batch_size >= 5e5: break

    mix_images_weights = np.concatenate(mix_images_weights, axis=0)
    unc_tags = np.concatenate(unc_tags, axis=0)

    x_oris_1_array = np.concatenate(x_oris_1_array, axis=0)
    x_oris_2_array = np.concatenate(x_oris_2_array, axis=0)

    mix_img_weights = []
    x_unlabeled_oris_1 = []
    x_unlabeled_oris_2 = []
    labeled_indices = []

    local_indices = np.argsort(unc_tags)[:10000]  #### TOP 10K are selected
    mix_img_weights.append(mix_images_weights[local_indices])
    labeled_indices.append(local_indices)
    mix_img_weights = np.concatenate(mix_img_weights, axis=0)
    labeled_indices = np.concatenate(labeled_indices, axis=0)

    k = 0
    for j in labeled_indices:
        mix_image.append(
            train_dataset[x_oris_1_array[j]][0].data.cpu().numpy()[:][:][:] * mix_img_weights[k] + 
            train_dataset[x_oris_2_array[j]][0].data.cpu().numpy()[:][:][:] * (1 - mix_img_weights[k]))
        k += 1
    mix_image = np.stack(mix_image)

    for j in range(len(x_oris_1)):
        if j not in labeled_indices:
            x_unlabeled_oris_1.append(x_oris_1[j])
            x_unlabeled_oris_2.append(x_oris_2[j])

    x_oris_1 = x_unlabeled_oris_1
    x_oris_2 = x_unlabeled_oris_2

    logger.info('Circle time: {} \t Number of mixup images: {} \t Number of left mixup images: {} and {}'.format(
        t, len(mix_image), len(x_unlabeled_oris_1), len(x_unlabeled_oris_2)
    ))


    ################ select mixup images using activaet learning ######################
    mix_array = []
    y_labeled_label_new = []
    y_labeled_logits_new = []
    val_loader = DataLoader(ArrayDataset(mix_image), batch_size=args.batch_size, shuffle=False)

    for img, _, _ in val_loader:
        img = img.type(torch.FloatTensor).to(device)

        logits = teacher(img)['output']
        probs = F.softmax(logits, dim=-1)
        target = torch.max(probs, dim=1)[1]

        mix_array.append(img.data.cpu().numpy())
        y_labeled_label_new.append(target.data.cpu().numpy())
        y_labeled_logits_new.append(probs.data.cpu().numpy())

    mix_array = np.concatenate(mix_array, axis=0)
    y_labeled_label_new = np.concatenate(y_labeled_label_new, axis=0)
    y_labeled_logits_new = np.concatenate(y_labeled_logits_new, axis=0)

    x_labeled_array = np.concatenate([mix_array, x_labeled_array], axis=0)
    y_labeled_label = np.concatenate([y_labeled_label_new, y_labeled_label], axis=0)
    y_labeled_logits = np.concatenate([y_labeled_logits_new, y_labeled_logits], axis=0)

    logger.info('Circle time: {} \t Total number of mixup and real images: {} \t Number of labels: {} \t Number of logits: {}'.format(
        t, len(x_labeled_array), len(y_labeled_label), len(y_labeled_logits)
    ))

    ################ train student model using mixup and real images ######################
    train()


logger.info('Max Test Accuracy : {}'.format(max_acc))
