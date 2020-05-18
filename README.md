# Multi-Neural-Gas

[Multi-Neural-Gas](https://en.wikipedia.org/wiki/Neural_gas), with Guassian as neighborhoodfunction, a rectangular grid and equally distributed random points drwan from the unit cube as weights.
Implemented in python, for a programming assignment of course ***Technical Neuroal Network***.
 
# Using Instruction

Simply import function "MultiNeuralGas" from MultiNeuralGas.py

	$ python
	from MultiNeuralGas import MultiNeuralGas
	
Now you can use this function with the following definition:

	Centers = MultiNeuralGas(M, N, K, Z0, Zend, Width, PartnerSizes, TrainingPatterns, MaxStep, RandomSeed)

### N, M, K

Number of partner networks, input dims, and gas neurons.

### Z0, Zend
Learning rule at the first and last iterations, to implement an exponentially decaying learning rate, decaying from Z0 to Zend.

### Width

The width of the Gaussian functions.

### PartnerSizes

A list consists of M positive integer values as the number of neurons for each of the partner networks(a total of K Neurons)

### TrainingPatterns 

A list of P patterns in the form of a list containing N real values as coordinators, 
**Alternatively**, the direction of a .dat file in which for each training pattern you must put the coordinate values in order followed by next patterns (lines with # consider as a comment).

### MaxSteps

The maximum number of iterations in which the model can train.

### RandomSeed

A random seed used for random initializing and shuffling, to be able to reproduce results.

### Centers

A list containing K lists as the coordination for each neuron’s center.

# Example:

*Please check "MultiNeuralGas-Test.py" for an example.*

![See Training-Patterns.png for Training Patterns of the example plotted using matplotlib](/Training-Patterns.png)
![See CLR.png to see the result of using a constant learning rate](/CLR.png)
![See DLR.png to see the result of using an exponentially decaying learning rate](/DLR.png)

# Authors 

	Ali Mohammadi
	Rozhin Bayati


*Best Regards*