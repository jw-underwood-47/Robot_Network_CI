# Decentralized learning with automated stepsizes
To run an experiment with neural networks, use
'''
python3 main.py
'''
while that folder.
For matrix factorization, use
'''
python3 mf.py
'''
and for logistic regression use
'''
python3 train.py
'''

Command line arguments are -s for stratified data and -g to control how often output is printed to the command window for all programs. -a and -p can be used to terminate the program on a given training and test accuracy, respectively, for neural networks (which tend to be the slowest).  -e (neural networks) and -i (the other two) can also be used to shorten the training and thus the program's runtime. -t and -r do not affect how the code behaves, but may be used in scripts and affect file names. -k is used to control kappa when using DOAS (not an option for neural networks).

I am working on bringing the three folders closer together in organization/style, as each currently uses different files and classes in different ways (the two train.py files, where certain logging functions are, etc.).


### original README
Each of the three experiments contain a `trainer.sh` script that can be used to train all runs for the given experiment, which will store all data in corresponding results folders with `.csv` files.

To run an experiment, use command:
```
./trainer.sh
```

### Dependencies

```
torch==1.10.2
torchvision==0.11.3
numpy==1.21.5
scipy==1.8.0
pandas==1.4.1
sklearn==1.0.2
matplotlib==3.5.1
```
