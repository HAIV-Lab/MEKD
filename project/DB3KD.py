import os
import time
import torch
from tools import Logger
from build import build_model, build_dataset, build_teacher
import argparse
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from numpy import linalg as LA


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

parser.add_argument('--t', type=float, default=20.0, help='Temperature')

parser.add_argument('--interval', type=int, default=100, help='Interval')
parser.add_argument('--save_model', type=str, default='./output/DB3KD/', help='Path to save model')
args = parser.parse_args()
args.save_model = os.path.join(args.save_model, args.dataset, '{}_to_{}'.format(args.teacher, args.model))


logger = Logger(os.path.join(args.save_model, '{}.log'.format(time.strftime('%Y%m%d%H%M%S', time.localtime()))))
logger.info(args)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_loader, test_loader = build_dataset(args)
train_dataset = train_loader.dataset
model = build_model(args).to(device)
teacher = build_teacher(args).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
KLDivLoss = torch.nn.KLDivLoss(reduction='sum').to(device)


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


class SampleDist(object):
    def __init__(self, train_dataset=None):
        self.train_dataset = train_dataset 

    def get_norm(self, x):
        return LA.norm(x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3]), axis=-1)

    def get_distance(self, x0, y0):
        batch_size = y0.size(0)

        g_thetas = np.array([float('inf') for _ in range(batch_size)])
        for i, (xi, yi) in enumerate(self.train_dataset):
            thetas = xi.cpu().numpy() - x0.cpu().numpy()
            lbds = self.get_norm(thetas)

            update_idx = np.where(lbds < g_thetas)[0]
            g_thetas[update_idx] = lbds[update_idx]

            if i >= 20:
                break
        return [g_thetas]


MAX_ITER = 1000
RANDN_TIME = 0.0

class MinimalBoundaryDist(object):
    def __init__(self, model, k=200, train_dataset=None, mode='mbd'):
        self.model = model
        self.k = k
        self.train_dataset = train_dataset 
        self.log = torch.ones(MAX_ITER, 2)
        self.mode = mode

    def get_log(self):
        return self.log

    def get_norm(self, x):
        return LA.norm(x.reshape(x.shape[0],x.shape[1]*x.shape[2]*x.shape[3]), axis=-1)


    def get_distance(self, x0, y0, target, alpha = 0.2, beta = 0.001, iterations = 5000, query_limit=20000,
                        query_idx = [1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000],
                        distortion=None, seed=None, svm=False, stopping=0.0001):
        """ Attack the original image and return adversarial example
            model: (pytorch model)
            train_dataset: set of training data
            (x0, y0): original image
        """    
        model = self.model
        batch_size = y0.size(0)

        query_counts = [0 for _ in range(batch_size)]
        best_thetas, g_thetas = np.zeros(x0.shape), np.array([float('inf') for _ in range(batch_size)])
        dists = []
        query_flags = [True for _ in range(len(query_idx))]

        timestart = time.time()

        # Iterate through training dataset. Find best initial point for gradient descent.
        for i, (xi, yi) in enumerate(self.train_dataset):
            thetas = xi.cpu().numpy() - x0.cpu().numpy() #[batch_size, 1, 28, 28], one theta for every sample
            initial_lbds = self.get_norm(thetas)

            thetas = thetas / initial_lbds[:, np.newaxis, np.newaxis, np.newaxis]

            lbds, counts = self.fine_grained_binary_search_targeted(model, x0, y0, target, thetas, initial_lbds, g_thetas)
            query_counts = [query_counts[i]+counts[i] for i in range(len(query_counts))]

            update_idx = np.where(lbds < g_thetas)[0]

            g_thetas[update_idx] = lbds[update_idx]

            best_thetas[update_idx,:,:,:] = thetas[update_idx,:,:,:]
            # if i >= 200:
            if i >= 10:
                break

        # timeend = time.time()

        # logger.info("==========> Found best average distance %.4f in %.4f seconds using %.2f mean queries" %
        #       (np.mean(g_thetas), timeend-timestart, np.mean(query_counts)))

        xg, gg = best_thetas, g_thetas
        dists.append(np.copy(gg))
        if self.mode == 'bd':
            return dists
        
        distortions = [gg]
        for i in range(iterations):
            t0=time.time()
            sign_gradients, grad_queries = self.sign_grad_v1(x0, y0, target, xg, initial_lbd=gg, h=beta)

            # Line search
            ls_counts = np.array([0 for _ in range(y0.size(0))])
            min_thetas = xg
            min_g2s = gg


            for _ in range(15):
                new_thetas = xg - alpha * sign_gradients

                new_thetas_mag = LA.norm(new_thetas.reshape(new_thetas.shape[0],new_thetas.shape[1]*new_thetas.shape[2]*new_thetas.shape[3]), axis=-1)
                new_thetas = new_thetas.reshape(new_thetas.shape[0],new_thetas.shape[1]*new_thetas.shape[2]*new_thetas.shape[3]) / new_thetas_mag[:,np.newaxis]
                new_thetas = new_thetas.reshape(xg.shape)

                new_g2s, counts = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, new_thetas, initial_lbd = min_g2s, tol=beta/500)
                ls_counts += counts
                alpha = alpha * 2

                new_g2s_small_idx = np.where(new_g2s < min_g2s)[0]

                if len(new_g2s_small_idx) > 0:
                    min_thetas[new_g2s_small_idx] = new_thetas[new_g2s_small_idx]
                    min_g2s[new_g2s_small_idx] = new_g2s[new_g2s_small_idx]
                else:
                    break
            
            min_g2s_bigger_idx = np.where(min_g2s >= gg)[0]

            if len(min_g2s_bigger_idx) != 0 and len(min_g2s_bigger_idx) != len(min_g2s):
                logger.info('Error on if min_g2 >= gg')
                exit()

            if len(min_g2s_bigger_idx) == len(min_g2s):
                for _ in range(15):
                    alpha = alpha * 0.25
                    new_thetas = xg - alpha * sign_gradients
                    new_thetas_mag = LA.norm(new_thetas.reshape(new_thetas.shape[0],new_thetas.shape[1]*new_thetas.shape[2]*new_thetas.shape[3]), axis=-1)
                    new_thetas = new_thetas.reshape(new_thetas.shape[0],new_thetas.shape[1]*new_thetas.shape[2]*new_thetas.shape[3]) / new_thetas_mag[:,np.newaxis]
                    new_thetas = new_thetas.reshape(xg.shape)

                    new_g2s, counts = self.fine_grained_binary_search_local_targeted(model, x0, y0, target, new_thetas, initial_lbd = min_g2s, tol=beta/500)
                    ls_counts += counts

                    new_g2s_small_idx = np.where(new_g2s < gg)[0]

                    if len(new_g2s_small_idx) > 0:
                        min_thetas[new_g2s_small_idx] = new_thetas[new_g2s_small_idx]
                        min_g2s[new_g2s_small_idx] = new_g2s[new_g2s_small_idx]
                        break

            if alpha < 1e-4:
                alpha = 1.0
                logger.info("Warning: not moving")
                beta = beta*0.1
                if (beta < 1e-8):
                    break

            xg, gg = min_thetas, min_g2s
            query_counts += (grad_queries + ls_counts)
            distortions.append(gg)

            for qi in range(len(query_idx)):
                if np.mean(query_counts) >= query_idx[qi] and query_flags[qi] == True:
                    dists.append(np.copy(gg))
                    query_flags[qi] = False
            if np.mean(query_counts) > query_limit:
                break

            if i%5==0:
                logger.info("Iteration %3d distance %.4f num_queries %d" % (i+1, np.mean(gg), np.mean(query_counts)))

        return dists

    def fine_grained_binary_search_local_targeted(self, model, x0, y0, t, theta, initial_lbd=1.0, tol=1e-5):
        nquerys = np.array([0 for _ in range(y0.size(0))])
        lbds = initial_lbd
        lbd_los = np.zeros_like(initial_lbd)
        lbd_his = np.copy(initial_lbd)

        adds = theta * lbds[:, np.newaxis, np.newaxis, np.newaxis]
        results = model.predict_label(x0 + torch.tensor(adds, dtype=torch.float).cuda(), batch=True)
        nquerys = [x+1 for x in nquerys]

        not_target_idx = np.where(results.cpu() != t)[0]
        target_idx = np.where(results.cpu() == t)[0]

        if len(not_target_idx) > 0:
            lbd_his[not_target_idx] = lbds[not_target_idx] * 1.01
            current_not_target_idx = np.copy(not_target_idx)

            while len(current_not_target_idx) > 0:
                add_his = theta * lbd_his[:, np.newaxis, np.newaxis, np.newaxis]
                results = model.predict_label(x0[current_not_target_idx] + torch.tensor(add_his[current_not_target_idx], dtype=torch.float).cuda(), batch=True)
                for i in current_not_target_idx:
                    nquerys[i] += 1
                current_not_target_idx = current_not_target_idx[np.where(results.cpu() != t)[0]]
                if len(current_not_target_idx) > 0:
                    lbd_his[current_not_target_idx] = lbd_his[current_not_target_idx] * 1.01
                    g100_idx = current_not_target_idx[np.where(lbd_his[current_not_target_idx] > 100)[0]]
                    if len(g100_idx) > 0:
                        current_not_target_idx = np.setdiff1d(current_not_target_idx,g100_idx)

        if len(target_idx) > 0:
            lbd_los[target_idx] = lbds[target_idx] * 0.99
            current_target_idx = np.copy(target_idx)
            while True:
                add_los = theta * lbd_los[:, np.newaxis, np.newaxis, np.newaxis]
                results = model.predict_label(x0[current_target_idx] + torch.tensor(add_los[current_target_idx], dtype=torch.float).cuda(), batch=True)
                for i in current_target_idx:
                    nquerys[i] += 1
                current_target_idx = current_target_idx[np.where(results.cpu() == t)[0]]
                if len(current_target_idx) > 0:
                    lbd_los[current_target_idx] = lbd_los[current_target_idx] * 0.99
                else:
                    break

        lbd_mids = np.zeros_like(lbd_his)
        need_binary_idx = np.where((lbd_his - lbd_los) > tol)[0]

        while len(need_binary_idx) > 0:
            lbd_mids[need_binary_idx] = (lbd_los[need_binary_idx] + lbd_his[need_binary_idx]) / 2.0
            
            for i in need_binary_idx:
                nquerys[i] += 1
            add_mids = theta * lbd_mids[:,np.newaxis, np.newaxis, np.newaxis]
            results = model.predict_label(x0[need_binary_idx] + torch.tensor(add_mids[need_binary_idx], dtype=torch.float).cuda(), batch=True)
            
            not_target_idx = need_binary_idx[np.where(results.cpu() != t)[0]]
            target_idx = need_binary_idx[np.where(results.cpu() == t)[0]]

            if len(target_idx) > 0:
                lbd_his[target_idx] = lbd_mids[target_idx]
            if len(not_target_idx) > 0:
                lbd_los[not_target_idx] = lbd_mids[not_target_idx]

            need_binary_idx = np.where((lbd_his - lbd_los) > tol)[0]

        return lbd_his, nquerys

    def sign_grad_v1(self, x0, y0, target, theta, initial_lbd, h=0.001, D=4):
        """
        Evaluate the sign of gradient by formulat
        $sign(g) = 1/Q [ \sum_{q=1}^Q sign( g(theta+h*u_i) - g(theta) )u_i$ ]$
        """
        global RANDN_TIME
        K = self.k
        sign_grad = np.zeros(theta.shape)   #[batch_size,1,28,28]
        queries = [0 for _ in range(y0.size(0))]
        # preds = []

        for iii in range(K):
            t00=time.time()

            u = torch.randn(*theta.shape).cpu().numpy()
            RANDN_TIME = RANDN_TIME + time.time() - t00
            u_mag = LA.norm(u.reshape(u.shape[0],u.shape[1]*u.shape[2]*u.shape[3]), axis=-1)
            u = u.reshape(u.shape[0],u.shape[1]*u.shape[2]*u.shape[3]) / u_mag[:,np.newaxis] * h
            u = u.reshape(theta.shape)

            signs = np.array([1 for _ in range(y0.size(0))])

            new_theta = theta + u
            new_theta_mag = LA.norm(new_theta.reshape(new_theta.shape[0],new_theta.shape[1]*new_theta.shape[2]*new_theta.shape[3]), axis=-1)
            new_theta = new_theta.reshape(new_theta.shape[0],new_theta.shape[1]*new_theta.shape[2]*new_theta.shape[3]) / new_theta_mag[:,np.newaxis]
            new_theta = new_theta.reshape(theta.shape)
            
            new_adds = new_theta.reshape(new_theta.shape[0],new_theta.shape[1]*new_theta.shape[2]*new_theta.shape[3]) * np.array(initial_lbd)[:,np.newaxis]
            new_adds = new_adds.reshape(theta.shape)

            results = self.model.predict_label(x0+torch.tensor(new_adds, dtype=torch.float).cuda(),batch=True)
            

            reverse_sign_idx = np.where(results.cpu() == target)[0]
            signs[reverse_sign_idx] = -1
            
            queries = [x+1 for x in queries]

            u = u.reshape(u.shape[0],u.shape[1]*u.shape[2]*u.shape[3]) * signs[:,np.newaxis]
            u = u.reshape(theta.shape)

            sign_grad += u

        return sign_grad, queries

    def fine_grained_binary_search_targeted(self, model, x0, y0, t, theta, initial_lbd, current_best):
        nquerys = [0 for _ in range(y0.size(0))]

        need_query_idx = np.where(initial_lbd > current_best)[0]
        no_need_query = []

        if need_query_idx.size != 0:
            need_query_adds = theta.reshape(theta.shape[0],theta.shape[1]*theta.shape[2]*theta.shape[3]) * np.array(current_best)[:,np.newaxis]
            need_query_adds = need_query_adds.reshape(theta.shape)[need_query_idx]
            results = model.predict_label(x0[need_query_idx] + torch.tensor(need_query_adds, dtype=torch.float).cuda(),batch=True)
            no_need_query = need_query_idx[np.where(results.cpu() != t)[0]]
            for i in need_query_idx:
                nquerys[i] += 1

        lbd = [current_best[i] if i in no_need_query else initial_lbd[i] for i in range(y0.size(0))]


        lbd_hi = np.array(lbd)
        lbd_lo = np.array([0.0 for _ in range(y0.size(0))])


        buffer = 0  # add to avoid dead loop
        while len(no_need_query) < y0.size(0):
            need_query = np.delete(list(range(y0.size(0))), np.array(no_need_query).astype(int))

            lbd_mid = np.array([(lbd_lo[i] + lbd_hi[i])/2.0 for i in range(y0.size(0))])
            for i in need_query:
                nquerys[i] += 1
            need_query_adds = theta.reshape(theta.shape[0],theta.shape[1]*theta.shape[2]*theta.shape[3])[need_query] * np.array(lbd_mid[need_query])[:,np.newaxis]
            need_query_adds =  need_query_adds.reshape(len(need_query), theta.shape[1], theta.shape[2], theta.shape[3])
            results = model.predict_label(x0[need_query] + torch.tensor(need_query_adds, dtype=torch.float).cuda(),batch=True)

            change_lo_idx = need_query[np.where(results.cpu() != t)[0]]
            change_hi_idx = need_query[np.where(results.cpu() == t)[0]]

            if len(change_lo_idx) > 0:
                lbd_lo[change_lo_idx] = lbd_mid[change_lo_idx]
            if len(change_hi_idx) > 0:
                lbd_hi[change_hi_idx] = lbd_mid[change_hi_idx]
            
            need_delete = need_query[np.where(np.array([(lbd_hi[i] - lbd_lo[i]) < 1e-5 for i in need_query]) == True)[0]]
            no_need_query = np.concatenate((no_need_query,need_delete))

            if len(need_delete) == 0: buffer += 1  # add to avoid dead loop
            else: buffer = 0  # add to avoid dead loop
            if buffer > 1000: break  # add to avoid dead loop

        return lbd_hi, nquerys


class ModelWrapper(object):
    def __init__(self, model, bounds, num_classes):
        self.model = model
        self.model.eval()
        self.bounds = bounds
        self.num_classes = num_classes
        self.num_queries = 0
    
    def predict(self,image):
        image = torch.clamp(image, self.bounds[0], self.bounds[1]).cuda()
        if len(image.size())!=4:
            image = image.unsqueeze(0)
        output = self.model(image)['output']
        self.num_queries += 1
        return output
 
    def predict_prob(self,image):
        with torch.no_grad():
            image = torch.clamp(image,self.bounds[0],self.bounds[1]).cuda()
            if len(image.size())!=4:
                image = image.unsqueeze(0)
            output = self.model(image)['output']
            self.num_queries += image.size(0)
        return output
    
    def predict_label(self, image, batch=False):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).type(torch.FloatTensor)
        image = torch.clamp(image,self.bounds[0],self.bounds[1]).cuda()
        if len(image.size())!=4:
            image = image.unsqueeze(0)

        with torch.no_grad():
            output = self.model(image)['output']
            self.num_queries += image.size(0)

        _, predict = torch.max(output.data, 1)
        if batch:
            return predict
        else:
            return predict[0]
    
    def predict_ensemble(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).type(torch.FloatTensor)
        image = torch.clamp(image,self.bounds[0],self.bounds[1]).cuda()
        if len(image.size())!=4:
            image = image.unsqueeze(0)
        with torch.no_grad():
            output = self.model(image)['output']
            output.zero_()
            for i in range(10):
                output += self.model(image)['output']
                self.num_queries += image.size(0)

        _, predict = torch.max(output.data, 1)
        
        return predict[0]

    def get_num_queries(self):
        return self.num_queries

    def get_gradient(self,loss):
        loss.backward()


def get_target_train_loader(target, batch_size, if_shuffle):
    return DataLoader(sub_train_set[target], batch_size=batch_size, shuffle=if_shuffle, num_workers=8)


sub_train_loader = []
sub_train_set = []
allidx = []
for i in range(args.num_classes):
    idx = np.where(np.array(train_dataset.labels) == i)[0].tolist()
    allidx.extend(idx)
    subset_one_class = torch.utils.data.dataset.Subset(train_dataset, idx)
    sub_train_set.append(subset_one_class)
    sub_train_loader.append(DataLoader(subset_one_class, batch_size=200, shuffle=False))

wrapped_model = ModelWrapper(teacher, bounds=[0,1], num_classes=args.num_classes)

logits_idx = ['sample_distance']
all_soft_labels = [[] for _ in range(len(logits_idx))]
for class_idx in range(0, len(sub_train_loader)):
    logger.info('Generating soft labels for samples from Class {}'.format(class_idx))
    targets = list(range(args.num_classes))
    targets.remove(class_idx)
    for i, (xi, yi) in enumerate(sub_train_loader[class_idx]):
        xi, yi = xi.cuda(), yi.cuda()

        logits = [np.zeros((xi.shape[0], args.num_classes)) for _ in range(len(logits_idx))]
        for target in targets:
            sr_method = MinimalBoundaryDist(wrapped_model, train_dataset=get_target_train_loader(target=target, batch_size=1, if_shuffle=True), mode = 'bd')
            dist = sr_method.get_distance(xi, yi, target)
            # sr_method = SampleDist(train_dataset=get_target_train_loader(target=target, batch_size=1, if_shuffle=True))
            # dist = sr_method.get_distance(xi, yi)
            
            assert len(dist)==len(logits_idx)
            for iii in range(len(logits_idx)):
                logits[iii][:, target] = 1. / dist[iii]
        
        for iii in range(len(logits_idx)):
            logits[iii][:, class_idx] = np.sum(logits[iii], axis=1)
            all_soft_labels[iii].extend(logits[iii].tolist())

all_soft_labels = np.array(all_soft_labels[0])


class ArrayDataset(Dataset):
    def __init__(self, ori_dataset, samples_idx, soft_labels):
        self.ori_dataset = ori_dataset
        self.samples_idx = samples_idx
        self.soft_labels = soft_labels
    
    def __getitem__(self, index):
        img = self.ori_dataset[self.samples_idx[index]][0]
        soft_label = self.soft_labels[index]
        return img, soft_label

    def __len__(self):
        return len(self.samples_idx)

train_loader = DataLoader(ArrayDataset(train_dataset, allidx, all_soft_labels), batch_size=args.batch_size, shuffle=True)


max_acc = 0
t = args.t
for epoch in range(args.epochs):
    model.train()
    for i, (data, soft_labels) in enumerate(train_loader):
        data = data.type(torch.FloatTensor).to(device)
        soft_labels = soft_labels.type(torch.FloatTensor).to(device)

        optimizer.zero_grad()
        output = model(data)["output"]
        loss = KLDivLoss(
            F.log_softmax(output / t, dim=-1),
            F.softmax(soft_labels, dim=-1)
        )
        loss.backward()
        optimizer.step()

        if i % args.interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] \t KL_loss: {:.6f}'.format(
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
