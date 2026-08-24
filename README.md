# Decentralized learning with automated stepsizes
To run an experiment with neural networks, logistic regression, or matrix factorization, use
```
python3 main.py [options]
```
while in that folder.  Results will be stored in `.csv` files.  

Command line arguments are as follows:
###### For all programs:
* -s or --stratified for stratified data
* -g or --print_gap to control how often output is printed to the command window  
* -k or --kappa to set kappa for functions that require it (DOAS/DLAS)  
  
###### For neural networks:
 * -a or --accuracy to terminate when a given training accuracy is achieved  
 * -p or --test_accuracy to terminate the program when a given test accuracy it achieved  
 * -e or --epochs to terminate training after the given number of epochs.
 Defaults to 800, which will take a while to run.  
 * -l or --learning_rate to set the learning rate (default 0.02).  
 * -b or --batch_size to set batch size (default 32)  
 * -t or --test_num to choose which trainer to use.  Default 0 (DAdSGD)  
 * -r or --run_num sets the run number, but should not effect any model
 behavior.
###### Other options
 * -i (valid for Logistic Regression and Matrix Factorization versions of main.py) terminates the program after a set number of iterations.  This can be used to reduce runtime.  
 * -t is also valid in Matrix-Factorization.

### Auto-launcher
In the Neural-Network folder there is a C file called `claude_fix_first_try.c`. To use this, compile it with ```gcc -o [exceutable_name] claude_fix_first_try.c```. This file may be renamed later.  
Arguments are similar to those for an individual run, but you can specify the minimum and maximum learning rates, number of epochs, and batch sizes by giving two arguments to those functions.  Giving one argument will cause all runs launched by the program to have that constant value.  The increment in number of epochs between runs defaults to 10, though adding a third number after -e allows you to change this.  I intend to add more useful arguments in the future.  The -h option provides more useful information  
This program uses fork() and exec() to launch the python scripts, so you still need to be in an environment where all of the dependencies are satisfied before running this code.
Additionally, running multiple tests in parallel is supported with the -j (or --jobs) option, which keeps track of how many separate instances of main.py are running and caps them at the set number (one by default).  This part was written by Claude (hence the name), and does not have some potentially desirable features such as in some way distinguishing what output goes with what main.py run.

### from original README:
Each of the three experiments contain a `trainer.sh` script that can be used to train all runs for the given experiment, which will store all data in corresponding results folders with `.csv` files.

To run an experiment, use command:
```
./trainer.sh
```

# Dependencies

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
