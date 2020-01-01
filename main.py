from graphTrain import *
from sceneEval import *
from sceneTrain import *
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument("--fromIdx", type=int, default=0,
                        help="First dataset to do")
    parser.add_argument("--toIdx", type=int, default=0,
                        help="Last dataset to do")
    parser.add_argument("--iters", type=int, default=50,
                        help="Number of total bayesian opt iterations")
    parser.add_argument("--root", type=str, default="data/",
                        help="Path to database")
    parser.add_argument("--save", type=str, default="checkpoints/",
                        help="Path to save to")
    parser.add_argument("--evalScene", action="store_true", default=False,
                        help="Evaluate classifiers on the scene datasets")
    parser.add_argument("--trainScene", action="store_true", default=False,
                        help="Evaluate classifiers on the scene datasets")

    args = parser.parse_args()

    if args.evalScene:
        evalScenes(args.root,args.save)
        exit(0)
    elif args.trainScene:
        trainScenes(args.root,args.save)
        exit(0)

    trainer = Trainer(args.root,args.save, args.iters)

    if args.fromIdx < 0:
        print("Warning: fromIdx must be at least 0")
    if args.toIdx > len(trainer):
        print("Warning: toIdx must be at most the length of the trainer")

    indices = [i for i in range(args.fromIdx,args.toIdx)]
    if len(indices) == 0:
        indices = [i for i in range(len(trainer))]

    trainer.trainModels(indices)
