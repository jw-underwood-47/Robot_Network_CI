# Decentralized learning with automated stepsizes
To run an experiment with neural networks, logistic regression, or matrix factorization, use
```
python3 main.py [options]
```
while in that folder.

Command line arguments are -s for stratified data and -g to control how often output is printed to the command window for all programs. -a and -p can be used to terminate the program on a given training and test accuracy, respectively, for neural networks (which tend to be the slowest).  -e (neural networks) and -i (the other two) can also be used to shorten the training and thus the program's runtime. -t and -r do not affect how the code behaves, but may be used in scripts and affect file names. -k is used to control kappa when using DOAS (not an option for neural networks).

I am working on bringing the three folders closer together in organization/style, as there are currently some differences in how class definitions are split across files (and the matrix factorization folder implements most of it as functions instead of methods).

### Auto-launcher
In the Neural-Network folder there is a C file called `run_many_times.c`. To use this, compile it with ```gcc -o [exceutable_name] run_many_times.c```.  
Arguments are similar to those for an individual run, but you can specify the minimum and maximum learning rates, number of epochs, and batch sizes by giving two arguments to those functions.  Giving one argument will cause all runs launched by the program to have that constant value.  The increment in number of epochs between runs defaults to 10, though adding a third number after -e allows you to change this.  I intend to add more useful arguments in the future.  The -h option provides more useful information  
This program uses system() to launch the python scripts, so you still need to be in an environment where all of the dependencies are satisfied before running this code.

### a small note about the use of system() in the launcher
I may fix this eventually, but currently the launcher uses system() to call the python script, using strings as given by the user.  This is not safe; running ```./[executable] -a 1; ls; # ``` would indeed cause the command ls to run once the rest of the program has completed (ignoring all non- -a options).

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
gcc to use the auto-launcher
```
