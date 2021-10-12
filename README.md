# FedComm
Experiments are produced on MNIST, CIFAR-10, and WikiText-2 datasets.

## Setting up the environment

* Install the requirements.
```
pip install -r requirements.txt
```

## Data
* Download the respective datasets and put them under 'data/' directory.

## Running the experiments

* To run the FedComm experiment:
```
python fedcomm.py
```

## Setting the experiment parameters
To run the experiments in different conditions change the parameters in *config.json* file.

#### Federated Learning Parameters
* ```num_users:```Number of total users that have signed up for collaborating. (Default is 100).
* ```frac:```     Fraction of users to be used for federated updates. Default is 1.0 (i.e., 100% participation).
* ```epochs:``` Number of global training epochs. Default is 1000.
* ```dataset:```  Default: 'mnist'. Options: 'mnist', 'cifar10', 'wiki'.

#### Message transmission parameters:

* ```senders:``` Fraction of participants in the federated learning scheme that will act as senders. Default: 0.1 (i.e., 10% of the participants). 
* ```payload:``` The extension of the payload file (under *payloads/* directory. Default 'txt', Options: 'txt', 'png'.
* ```injection:``` The FL global round when the senders should start transmitting the message. Default 10.
* ```stealthy:``` The level of stealthiness of the *senders*. Default 'non', Options: 'non', 'inter', 'full'.
* ```run_name:```  A name given to the particular run. It will create a directory structure where it will store model checkpoints, extracted payloads, train accuracy and loss values.

----

