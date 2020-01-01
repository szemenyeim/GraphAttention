from data import *
from model import *
import os
import csv
from torch.utils.data import DataLoader, SubsetRandomSampler
import random
import progressbar

def trainScenes(dataroot,saveroot):

    dirs = sorted([name for name in os.listdir(dataroot) if os.path.isdir(os.path.join(dataroot, name))],
                   key=alphanum_key)

    accuracies = []

    for dir in dirs:

        print("Dataset: ",dir)

        dataPath = osp.join(dataroot,dir)
        savePath = osp.join(saveroot,dir)

        # Reproducibility
        random_seed = 42
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        dataset = GraphDataSet(dataPath,scene=True)
        nClass = dataset.nClass.item()

        # Split dataset
        validation_split = .2
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        f = open(osp.join(savePath, "params.csv"), 'r')
        reader = csv.reader(f)
        params = dict()
        for key,value in reader:
            params[key] = float(value)
        f.close()

        bFeat = int(params['bFeatNum'])
        eFeat = int(params['eFeatNum'])
        numEpoch = int(params['numEpoch'])
        bSize = int(params['batch_size'])
        decay = params['decay']
        lr = params['lr']
        factor = params['factor']

        net = GraphClassifier(bFeat,eFeat,nClass).cuda()

        feat = torch.load(osp.join(savePath, "bestFeat.pth"))
        classifier = torch.load(osp.join(savePath, "bestClass.pth"))

        net.Features.load_state_dict(feat)
        net.Classifier.load_state_dict(classifier)

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(dataset, batch_size=bSize,
                                                   sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=bSize,
                                                  sampler=valid_sampler)

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
            #bar = progressbar.ProgressBar(0, len(train_loader), redirect_stdout=False)

            for i, (inputs, labels) in enumerate(train_loader, 0):
                inputs = [input.cuda() for input in inputs]
                labels = labels.cuda().view(-1)
                masks = torch.logical_not(inputs[-1].view(-1))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs[0], inputs[1], inputs[2], inputs[3]).view(-1, nClass)
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
            #print("Train epoch %d correct: %.2f" % (epoch + 1, tr_corr))
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
                outputs = net(inputs[0], inputs[1], inputs[2], inputs[3]).view(-1, nClass)
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
            print("Test epoch %d correct: %.2f" % (epoch + 1, val_corr))

            return val_loss, val_corr

        best_acc = torch.tensor([0]).cuda()

        for epoch in range(numEpoch):
            tr_loss, tr_acc = train(epoch)
            val_loss, val_acc = val(epoch)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(net.state_dict(), osp.join(savePath, "bestScene.pth"))

            # Step with the scheduler
            scheduler.step()

        accuracies.append(best_acc.item())

    np.savetxt("sceneRes.csv",np.array(accuracies),delimiter=',')