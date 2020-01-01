from data import *
from model import *
import os
import csv
import progressbar

def evalScenes(dataroot,saveroot):

    dirs = sorted([name for name in os.listdir(dataroot) if os.path.isdir(os.path.join(dataroot, name))],
                   key=alphanum_key)

    accuracies = []

    for dir in dirs:

        print("Dataset: ",dir)

        dataPath = osp.join(dataroot,dir)
        savePath = osp.join(saveroot,dir)

        dataset = GraphDataSet(dataPath,scene=True)
        nClass = dataset.nClass.item()

        f = open(osp.join(savePath, "params.csv"), 'r')
        reader = csv.reader(f)
        params = dict()
        for key,value in reader:
            params[key] = float(value)
        f.close()

        net = GraphClassifier(int(params['bFeatNum']),int(params['eFeatNum']),nClass).cuda()

        feat = torch.load(osp.join(savePath, "bestFeat.pth"))
        classifier = torch.load(osp.join(savePath, "bestClass.pth"))

        net.Features.load_state_dict(feat)
        net.Classifier.load_state_dict(classifier)

        test_loader = torch.utils.data.DataLoader(dataset, batch_size=int(params['batch_size']))

        criterion = nn.CrossEntropyLoss(reduction='none')

        def val():
            # variables for loss
            running_loss = 0.0
            correct = 0.0
            total = 0

            # set the network to eval (for batchnorm and dropout)
            net.eval()

            # Create progress bar
            bar = progressbar.ProgressBar(0, len(test_loader), redirect_stdout=False)

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
                bar.update(i)

            # Finish progress bar
            bar.finish()

            # print and plot statistics
            val_loss = running_loss / len(test_loader)
            val_corr = correct / total * 100
            print("Test loss: %.3f correct: %.2f" % (val_loss, val_corr))

            return val_loss, val_corr

        loss,corr = val()

        accuracies.append(corr.item())

    np.savetxt("sceneClassRes.csv",np.array(accuracies),delimiter=',')