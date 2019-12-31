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

global best_acc, savePath, table, iter, bestTotalAcc

def evalParams(batch_size, baseFeatNum, edgeFeatNum, lr, numEpoch, factor, decay):

    batch_size = int(pow(2,round(batch_size)))
    baseFeatNum = int(pow(2,round(baseFeatNum)))
    edgeFeatNum = int(pow(2,round(edgeFeatNum)))
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
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)

    net = GraphClassifier(baseFeatNum,edgeFeatNum).cuda()
    criterion = nn.CrossEntropyLoss(reduction='none')

    optimizer = torch.optim.Adam(net.parameters(),lr=lr, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, numEpoch, lr*factor)

    def train(epoch):

        # variables for loss
        running_loss = 0.0
        correct = 0.0
        total = 0

        # set the network to train (for batchnorm and dropout)
        net.train()

        # Create progress bar
        #bar = progressbar.ProgressBar(0, len(train_loader), redirect_stdout=False)

        for i, (inputs, labels) in enumerate(train_loader, 0):

            inputs = [input.cuda() for input in inputs]
            labels = labels.cuda().view(-1)
            masks = torch.logical_not(inputs[-1].view(-1))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs[0],inputs[1],inputs[2],inputs[3]).view(-1,5)
            loss = criterion(outputs, labels)[masks].mean()
            loss.backward()
            optimizer.step()

            # compute statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += masks.sum()
            correct += predicted[masks].eq(labels[masks]).sum()

            #bar.update(i)

        #bar.finish()
        # print and plot statistics
        tr_loss = running_loss / len(train_loader)
        tr_corr = correct / total * 100
        #print("Train epoch %d lr: %.3f loss: %.3f correct: %.2f" % (epoch + 1, scheduler.get_lr()[0]/lr, tr_loss, tr_corr))
        return tr_loss, tr_corr

    def val(epoch):
        # variables for loss
        running_loss = 0.0
        correct = 0.0
        total = 0

        # set the network to eval (for batchnorm and dropout)
        net.eval()

        # Create progress bar
        #bar = progressbar.ProgressBar(0, len(test_loader), redirect_stdout=False)

        # Epoch loop
        for i, (inputs, labels) in enumerate(test_loader, 0):

            inputs = [input.cuda() for input in inputs]
            labels = labels.cuda().view(-1)
            masks = torch.logical_not(inputs[-1].view(-1))

            # forward
            outputs = net(inputs[0],inputs[1],inputs[2],inputs[3]).view(-1,5)
            loss = criterion(outputs, labels)[masks].mean()

            # compute statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += masks.sum()
            correct += predicted[masks].eq(labels[masks]).sum()

            # Update progress bar
            #bar.update(i)

        # Finish progress bar
        #bar.finish()

        # print and plot statistics
        val_loss = running_loss / len(test_loader)
        val_corr = correct / total * 100
        #print("Test epoch %d loss: %.3f correct: %.2f" % (epoch + 1, val_loss, val_corr))

        return val_loss, val_corr

    trAccs = []
    trLosses = []
    valAccs = []
    valLosses = []
    x = range(numEpoch)

    global best_acc, savePath, table, iter, bestTotalAcc
    best_acc = torch.tensor([0]).cuda()
    iter += 1

    for epoch in range(numEpoch):
        tr_loss, tr_acc = train(epoch)
        val_loss, val_acc = val(epoch)

        trAccs.append(tr_acc)
        trLosses.append(tr_loss)
        valAccs.append(val_acc)
        valLosses.append(val_loss)

        if val_acc > best_acc:
            best_acc = val_acc
        if val_acc > bestTotalAcc:
            torch.save(net.Features.state_dict(),savePath+"bestClass.pth")

        # Step with the scheduler
        scheduler.step()

    '''plt.figure()
    plt.plot(x,trLosses,x,valLosses)
    plt.figure()
    plt.plot(x,trAccs,x,valAccs)
    plt.show()'''
    table.add_row([iter,best_acc.item(),batch_size,baseFeatNum,edgeFeatNum,lr,factor,decay,numEpoch])
    if best_acc > bestTotalAcc:
        bestTotalAcc = best_acc
        print(colored("\n".join(table.get_string().splitlines()[-2:-1]),'cyan'))
    else:
        print("\n".join(table.get_string().splitlines()[-2:-1]))
    print("\n".join(table.get_string().splitlines()[-1:]))
    return best_acc


if __name__ == '__main__':

    root = "data/"
    saveRoot = "checkpoints/"

    global best_acc, savePath, table, iter, bestTotalAcc

    accs = []

    # Bounded region of parameter space
    pbounds = {'batch_size': (3, 8), 'baseFeatNum': (4, 9), 'edgeFeatNum': (0, 5), 'lr': (1e-5, 1e-1),
               'numEpoch': (20, 50), 'factor': (1e-2, 1),  'decay': (1e-7, 1e-3), }

    for i in range(20):

        dir = ("s%d/" % (i+1))
        path = root+dir

        iter = 0

        bestTotalAcc = 0

        savePath = saveRoot+dir
        if not os.path.exists(savePath):
            os.makedirs(savePath)

        dataset = GraphDataSet(path)

        names = ['iter', 'accuracy', 'batchSize', 'features', 'edgeFeat', 'learning_rate', 'factor', 'weight_decay', 'epochs']
        table = PrettyTable(names)
        table.hrules = 1
        table.float_format['accuracy'] = '2.3'
        table.float_format['learning_rate'] = '.1e'
        table.float_format['factor'] = '0.2'
        table.float_format['weight_decay'] = '.1e'
        print( "\n".join(table.get_string().splitlines()) )

        optimizer = BayesianOptimization(
            f=evalParams,
            pbounds=pbounds,
            verbose=0,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=1,
        )

        optimizer.maximize(init_points=7, n_iter=43,)

        accs.append(optimizer.max)

    np.savetxt("res.csv",np.array(accs),fmt='%.3f',delimiter=',')