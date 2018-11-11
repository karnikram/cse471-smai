### Contents

[statement](./statement.md) - problem statement

[report](./report.pdf) - my write up

[code](./code) - contains code for the training/validation/testing of four models - Multi-layer Perceptron, Linear Support Vector Machine, Kernel Support Vector Machine, and Logistic regression - all implemented using scikit-learn.

[slurm-scripts](./slurm-scripts) - contains some simple slurm scripts that were used for running the models on the abacus campus cluster and for obtaining the results, and their outputs.

[plots](./plots) - contains a few plots that were generated for the write up

### Requirements
* Python 3.7
* scikit-learn
* numpy

### Running the code

```bash
cd ./code

# run mlp classifier with predetermined hyperparameters
python mlp --best

# search for best hidden layer dimensions of model
python mlp.py --search_lu 

# search for best learning rate with fixed hidden layer dimensions
python mlp.py --search_lr

# run simple example of overfitting
python mlp.py --overfit

# run mlp classifier after transforming samples with pca
python mlp.py --pca --best

# run mlp classifier after transforming samples with lda
python mlp.py --lda --best

# run mlp classifier after transforming samples with pca followed by lda
python mlp.py --pca --lda --best

# run mlp classifier after scaling samples
python mlp.py --scaling --best

# run mlp classifier after scaling and mean subtracting samples
python mlp.py --mean-sub --scaling --best

```

The other classifiers can be run similarly by changing the file name and search arguments.
