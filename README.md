# GraphAttention
Graph Attention: A simple attention mechanism for graph node classification

Requirements:

* Numpy
* Pytorch 1.x

To run the training, run `python main.py`. This will run Bayesian optimization + training on classification datasets.

To evaluate the trained model on the scene datasets, use `python main.py --evalScene`

To finetune a model on the scene datasets, use `python main.py --trainScene`
