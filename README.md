# Lab 3 Report

**Zhe ZHANG   zz3230**

```bash
├── data  // Dataset is not uploaded due to poor network condition
    └── cl
        └── valid.h5 // this is clean validation data used to design the defense
        └── test.h5  // this is clean test data used to evaluate the BadNet
    └── bd
        └── bd_valid.h5 // this is sunglasses poisoned validation data
        └── bd_test.h5  // this is sunglasses poisoned test data
├── models
    └── bd_net.h5
    └── bd_weights.h5
├── report //Pruned part of repaired network)
	└── pruning_channel_model_acc_decrease_by_2%.h5
    └── pruning_channel_model_acc_decrease_by_4%.h5
    └── pruning_channel_model_acc_decrease_by_10%.h5
    └── pruning_channel_model_acc_decrease_by_30%.h5  
├── detector.ipynb  // Source code
└── detector.pdf  // Results and figs of running Code
└── README.md  // Lab report
└── model.png  // Main structure of the backdoored network
└── ML_Security_.pdf  // Lab3 instruction and requirements
```

## I. Dependencies

I use **Anaconda** environment and **jupyter notebook** to design this repaired network as a detector to defend against **backdoor attack**, implementing **pruning channels of a specific layer**

   1. Python 3.7.9

   2. Keras 2.3.1

   3. Numpy 1.16.3

   4. Matplotlib 2.2.2

   5. H5py 2.9.0

   6. TensorFlow-gpu 1.15.2

      ...

## II. Data
   1. Download the validation and test datasets from [here](https://drive.google.com/drive/folders/1Rs68uH8Xqa4j6UxG53wzD0uyI8347dSq?usp=sharing) and store them under `data/` directory.
   2. The dataset contains images from YouTube Aligned Face Dataset. We retrieve 1283 individuals and split into validation and test datasets.
   3. bd_valid.h5 and bd_test.h5 contains validation and test images with sunglasses trigger respectively, that activates the backdoor for bd_net.h5. 

## III. Building a Repaired Network and Evaluation
   1. **All codes are in** *'./detector.ipynb'*, including building and evaluating methods . Just set up a jupyter host and **run it** !!!

   2. For the results and, please look through *'./detector.pdf'*

   3. Codes can be seperated into several parts:

      ***Prework***: import libraries, define data load & processing method, relating display methods

      

      ***Prune Channels***: prune the last pooling layer of BadNet B (the layer just before the FC layers) by removing one channel at a time from that layer. Channels should be removed in **increasing order** of average activation values over the entire validation set.

      

      ***Good Network***: Combine network $B$ and network $B^{'}$ to be the repaired network $G$

      

      ***Evaluation***: rewrite the eval.py as a method, compare the performance among **original backdoored network** and    **repaired  networks** on test dataset

      

## IV. Results and Figures
Please use only clean validation data (valid.h5) to design the pruning defense. And use test data (test.h5 and bd_test.h5) to evaluate the models. 

### 1.Repaired Network

#### a.  Find the channel to be pruned

***increasing*** order of activation values of **'the last pooling layer'** of BadNet B

```bash
Increasing order of average activation values
[0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00 0.0000000e+00
 0.0000000e+00 3.0290761e-03 6.2408661e-03 1.3321567e-02 1.5006668e-02
 4.3979675e-02 8.3539173e-02 1.8378231e-01 2.4381575e-01 4.2763174e-01
 5.0732863e-01 5.3079778e-01 5.7658589e-01 8.5787797e-01 1.0589781e+00
 1.5654888e+00 1.6352932e+00 1.8540378e+00 2.0289588e+00 2.1106052e+00
 2.1980376e+00 3.6191154e+00 4.1488924e+00 4.8440871e+00 4.8648086e+00
 5.0868411e+00 5.1451392e+00 5.3690357e+00 6.2038713e+00 8.2229824e+00]
```

#### b.  Find the channel to be pruned

Every time prune a channel, I measure the new validation accuracy of the new pruned BadNet. Just Stop pruning once the validation accuracy drops at least **X%** below the original accuracy.  

```python
The original accuracy is  98.64899974019225
```

```python
45 channels pruned, acc = 95.75647354291158, atk = 100.0， X = 2.8925261972806737
pruning channel model accuracy decrease by 2 saved at prune channels of 45.0
48 channels pruned, acc = 92.09318437689443, atk = 99.9913397419243， X = 6.555815363297825
pruning channel model accuracy decrease by 4 saved at prune channels of 48.0
52 channels pruned, acc = 84.43751623798389, atk = 77.015675067117， X = 14.211483502208367
pruning channel model accuracy decrease by 10 saved at prune channels of 52.0
54 channels pruned, acc = 54.8627349095003, atk = 6.954187234779596， X = 43.786264830691955
pruning channel model accuracy decrease by 30 saved at prune channels of 54.0
```

#### c. Repair the BadNet

For each test input, run it through both B and B'. If the classification outputs are the same, i.e., class $i$, the network will output class $i$ in **[0,N-1]**. If they differ, the network will output **N**

using the class below

```python
class good_network(keras.Model):
    
    def __init__(self, B, B_pruned):
        super(good_network,self).__init__()
        self.B = B
        self.B_pruned = B_pruned
         
    def predict(self, X):
        yhat = np.argmax(self.B.predict(X),axis=1)
        yhat_pruned = np.argmax(self.B_pruned.predict(X),axis=1)
        yhat[np.where(yhat!=yhat_pruned)] = 1283
        
        return yhat
```
four network at $X = [2, 4, 10, 30]$ in '*.h5' form can be found in route './result', they are just $B^{'}$ not the complete network

#### d. Performance

```python
backdoored network on test dataset
Clean Classification accuracy: 98.62042088854248
Attack Success Rate: 100.0
repaired network, channels pruned with a 2% drop of accuracy, on test dataset
Clean Classification accuracy: 95.74434918160561
Attack Success Rate: 100.0
repaired network, channels pruned with a 4% drop of accuracy, on test dataset
Clean Classification accuracy: 92.1278254091972
Attack Success Rate: 99.98441153546376
repaired network, channels pruned with a 10% drop of accuracy, on test dataset
Clean Classification accuracy: 84.3335931410756
Attack Success Rate: 77.20966484801247
repaired network, channels pruned with a 30% drop of accuracy, on test dataset
Clean Classification accuracy: 54.67653936087296
Attack Success Rate: 6.96024941543258
```

The performance of the repaired network is pretty good, especially at **X = 30**, the network can **recognize half of the test data correctly**, and **defense against most of backdoored attacks(94%)** successfully

### 2.Acc and ASR on test data

Plot the **accuracy (on clean test data)** and the **attack success rate (on backdoored test data)** as a function of the fraction of channels pruned  

![acc&asr](https://github.com/NYUzhangzhe/mlCyber-lab3/blob/main/index.png)

we can find that, the **origin acc** is 98.6% and **attack success rate** is 100% on test data.

However, as we prune the channels in a **significance-increasing** order, the attack success rate drop sharply at the channel pruned ratio of approximately **0.85**, at the time, **X = 30**. We get a **Clean Classification accuracy of 54.67%**
**Attack Success Rate of 6.96%**

I believe it is **not a bad defense**. In the area between blue and orange line, it is in a backdoor disabled status. In such circumstance, most of the neurons which are not strongly relating to the specific features in the classification area are pruned.  So, the misleading of the backdoored dataset are disabled.

In other words, this $B^{'}$model is much sparse than $B$. 

## V. Important Notes

1. Good Net is not a complete and functional model, has only one method *predict()* .  It runs as a subclass containing 2 models, I would try to make it a storable model in '.h5' or other format. One should first load the $B^{'}$ uses *keras.model_load()*, then create a class *good_net*  with $B$ and $B^{'}$
2. The eval.py is not available in this lab, I rewrite it in similar way. Also it is due to formation of repaired network. 
3. Continuing struggling in **FINEL_PROJECT** , good luck and have fun
