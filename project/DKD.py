import os
import time
import torch
from tools import Logger
from build import build_model, build_dataset, build_teacher
import argparse
import torch.nn.functional as F


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

parser.add_argument('--epochs', type=int, default=240, help='Epochs')
parser.add_argument('--milestones', type=int, nargs='+', default=[80, 160], help='Milestones')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma')
parser.add_argument('--lamda', type=float, default=0.1, help='Ratio')

parser.add_argument('--interval', type=int, default=100, help='Interval')
parser.add_argument('--save_model', type=str, default='./output/DKD/', help='Path to save model')
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


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher, size_average=False)
        * (temperature)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, size_average=False)
        * (temperature)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


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
    for i, (data, labels) in enumerate(train_loader):
        data = data.type(torch.FloatTensor).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.no_grad():
            teacher_output = teacher(data)['output']
        output = model(data)["output"]
        loss = dkd_loss(
            output, teacher_output, 
            teacher_output.argmax(dim=-1, keepdim=True).squeeze(),
            1.0, 8.0, 5.0
        )
        loss.backward()
        optimizer.step()

        if i % args.interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tCE_loss: {:.6f}'.format(
                epoch, i*len(data), len(train_loader.dataset), 100.*i/len(train_loader), loss.item()
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
