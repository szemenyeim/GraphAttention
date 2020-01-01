from data import *
from model import *
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
import matplotlib.pyplot as plt
import progressbar
import os
from bayes_opt import BayesianOptimization
from prettytable import PrettyTable
from termcolor import colored
import csv


class Trainer(object):
    def __init__(self,dataroot,saveroot,iters):

        self.dataroot = dataroot
        self.saveroot = saveroot

        self.dirs = sorted([name for name in os.listdir(dataroot) if os.path.isdir(os.path.join(dataroot, name))], key=alphanum_key)

        self.accs = np.zeros(len(self.dirs))

        self.best_acc = 0
        self.bestTotalAcc = 0

        self.iter = 0
        self.numClass = 0
        self.iters = iters

        self.savePath = ""
        self.dataset = None

        self.table = None
        self.pbounds = {'batch_size': (3, 8), 'bFeatNum': (4, 9), 'eFeatNum': (0, 5),
                        #'dropout': (0.0, 0.25),
                        'lr': (1e-4, 1e-1), 'numEpoch': (20, 50), 'factor': (1e-2, 1),  'decay': (1e-7, 1e-3), }


    def __len__(self):
        return len(self.dirs)

    def trainModels(self,indices):

        dirs = [self.dirs[i] for i in indices]

        for i,dir in enumerate(dirs):

            # dir = ("s%d/" % (i+1))
            #dir = "img/"
            path = self.dataroot + dir

            self.iter = 0

            self.bestTotalAcc = 0

            self.savePath = self.saveroot + dir
            if not os.path.exists(self.savePath):
                os.makedirs(self.savePath)

            self.dataset = GraphDataSet(path)
            self.numClass = self.dataset.nClass.item()

            names = ['iter', 'accuracy', 'bSize', 'bFeatures', 'eFeatures',
                     #'dropout',
                     'learning_rate', 'ratio', 'weight_decay', 'epochs']
            self.table = PrettyTable(names)
            self.table.title = 'Dataset: ' + dir
            self.table.hrules = 1
            self.table.float_format['accuracy'] = '2.3'
            self.table.float_format['learning_rate'] = '.1e'
            self.table.float_format['ratio'] = '0.2'
            #self.table.float_format['dropout'] = '0.2'
            self.table.float_format['weight_decay'] = '.1e'
            print("\n".join(self.table.get_string().splitlines()))

            optimizer = BayesianOptimization(
                f=self.evalParams,
                pbounds=self.pbounds,
                verbose=0,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
                random_state=1,
            )

            initCnt = min(self.iters,len(self.pbounds))
            nIter = self.iters-initCnt

            optimizer.maximize(init_points=initCnt, n_iter=nIter)

            paramFile = osp.join(self.savePath, "params.csv")
            f = open(paramFile, "w")
            w = csv.writer(f)
            params = optimizer.max['params']
            for key in params:
                val = params[key]
                if key in ['batch_size','bFeatNum','eFeatNum']:
                    val = int(pow(2, round(val)))
                elif key == 'numEpoch':
                    val = int(round(val))
                w.writerow([key, val])
            f.close()

            self.accs[indices[i]] = optimizer.max['target'].item()

        if osp.exists("results.csv"):
            accs = np.loadtxt("results.csv",delimiter=',')
        else:
            accs = np.zeros(len(self.dirs))
        accs[self.accs != 0] = self.accs[self.accs != 0]
        np.savetxt("results.csv", accs, fmt='%.3f', delimiter=',')

    def evalParams(self, batch_size, bFeatNum, eFeatNum, lr, numEpoch, factor, decay, dropout = 0):

        batch_size = int(pow(2, round(batch_size)))
        baseFeatNum = int(pow(2, round(bFeatNum)))
        edgeFeatNum = int(pow(2, round(eFeatNum)))
        numEpoch = int(round(numEpoch))

        # Reproducibility
        random_seed = 42
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Split dataset
        validation_split = .2
        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                                   sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size,
                                                  sampler=valid_sampler)

        net = GraphClassifier(baseFeatNum, edgeFeatNum, self.numClass, dropout).cuda()
        criterion = nn.CrossEntropyLoss(reduction='none')

        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, numEpoch, lr * factor)

        def train(epoch):

            # variables for loss
            running_loss = 0.0
            correct = 0.0
            total = 0

            # set the network to train (for batchnorm and dropout)
            net.train()

            # Create progress bar
            # bar = progressbar.ProgressBar(0, len(train_loader), redirect_stdout=False)

            for i, (inputs, labels) in enumerate(train_loader, 0):
                inputs = [input.cuda() for input in inputs]
                labels = labels.cuda().view(-1)
                masks = torch.logical_not(inputs[-1].view(-1))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs[0], inputs[1], inputs[2], inputs[3]).view(-1, self.numClass)
                loss = criterion(outputs, labels)[masks].mean()
                loss.backward()
                optimizer.step()

                # compute statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += masks.sum()
                correct += predicted[masks].eq(labels[masks]).sum()

                # bar.update(i)

            # bar.finish()
            # print and plot statistics
            tr_loss = running_loss / len(train_loader)
            tr_corr = correct / total * 100
            # print("Train epoch %d lr: %.3f loss: %.3f correct: %.2f" % (epoch + 1, scheduler.get_lr()[0]/lr, tr_loss, tr_corr))
            return tr_loss, tr_corr

        def val(epoch):
            # variables for loss
            running_loss = 0.0
            correct = 0.0
            total = 0

            # set the network to eval (for batchnorm and dropout)
            net.eval()

            # Create progress bar
            # bar = progressbar.ProgressBar(0, len(test_loader), redirect_stdout=False)

            # Epoch loop
            for i, (inputs, labels) in enumerate(test_loader, 0):
                inputs = [input.cuda() for input in inputs]
                labels = labels.cuda().view(-1)
                masks = torch.logical_not(inputs[-1].view(-1))

                # forward
                outputs = net(inputs[0], inputs[1], inputs[2], inputs[3]).view(-1, self.numClass)
                loss = criterion(outputs, labels)[masks].mean()

                # compute statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += masks.sum()
                correct += predicted[masks].eq(labels[masks]).sum()

                # Update progress bar
                # bar.update(i)

            # Finish progress bar
            # bar.finish()

            # print and plot statistics
            val_loss = running_loss / len(test_loader)
            val_corr = correct / total * 100
            # print("Test epoch %d loss: %.3f correct: %.2f" % (epoch + 1, val_loss, val_corr))

            return val_loss, val_corr

        best_acc = torch.tensor([0]).cuda()
        self.iter += 1

        for epoch in range(numEpoch):
            tr_loss, tr_acc = train(epoch)
            val_loss, val_acc = val(epoch)

            if val_acc > best_acc:
                best_acc = val_acc
            if val_acc > self.bestTotalAcc:
                torch.save(net.Features.state_dict(), osp.join(self.savePath, "bestFeat.pth"))
                torch.save(net.Classifier.state_dict(), osp.join(self.savePath, "bestClass.pth"))
                pass

            # Step with the scheduler
            scheduler.step()

        self.table.add_row([self.iter, best_acc.item(), batch_size, baseFeatNum, edgeFeatNum,
                            #dropout,
                            lr, factor, decay, numEpoch])
        if best_acc > self.bestTotalAcc:
            self.bestTotalAcc = best_acc
            print(colored("\n".join(self.table.get_string().splitlines()[-2:-1]), 'cyan'))
        else:
            print("\n".join(self.table.get_string().splitlines()[-2:-1]))
        print("\n".join(self.table.get_string().splitlines()[-1:]))

        return best_acc