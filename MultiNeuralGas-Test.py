import random as rand
from matplotlib import pyplot as plt
import math
from MultiNeuralGas import MultiNeuralGas

RandomSeed = math.pi
M = 4
N = 2
K = 30
Width = 1.5
PartnerSizes = []
Area1 = [[0.75 + rand.random() * 0.2, 0.75 + rand.random() * 0.2] for i in range(50)] # Square with center (0.75, 0.75) and Width 0.4
Area2 = [[-0.5 + rand.random() * 0.4, -0.5 + rand.random() * 0.25] for i in range(50)] # Rectangle with Center (-0.5, -0.5) and lengths 0.8, 0.5
Area3 = [[rand.random() * 0.1, rand.random() * 2 * math.pi] for i in range(50)]
Area3 = [[-0.5 + Point[0] * math.cos(Point[1]), 0.5 + Point[0] * math.sin(Point[1])] for Point in Area3] # Circle with Center (-0.5, 0.5) and R 0.1
TrainingPatterns = Area1 + Area2 + Area3 # A set of equally distributed points within 3 areas of input space with above-mentioned bounderies
TrainingPatterns_x = [p[0] for p in TrainingPatterns]
TrainingPatterns_y = [p[1] for p in TrainingPatterns]
plt.figure()
plt.plot(TrainingPatterns_x, TrainingPatterns_y, '.')
plt.title('Training Patterns')
plt.xlim([-1, 1])
plt.ylim([-1, 1])

Z0 = 0.4 #Learning rule at the beginning
Zend = 0.4 #Learning rule at the end
plt.figure()
for i, MaxStep in enumerate([0, 10, 100, 250, 500, 1000]):
    Centers = MultiNeuralGas(M, N, K, Z0, Zend, Width, PartnerSizes, TrainingPatterns, MaxStep, RandomSeed)
    Centers_x = [p[0] for p in Centers]
    Centers_y = [p[1] for p in Centers]
    plt.subplot(2, 3, i + 1)
    plt.plot(Centers_x, Centers_y, '.')
    plt.title('With ' + str(MaxStep) + ' iterations, Learning Rate: 0.2')
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])

Z0 = 0.9 #Learning rule at the beginning
Zend = 0.1 #Learning rule at the end
MaxStep = 250
Centers = MultiNeuralGas(M, N, K, Z0, Zend, Width, PartnerSizes, TrainingPatterns, MaxStep, RandomSeed)
Centers_x = [p[0] for p in Centers]
Centers_y = [p[1] for p in Centers]
plt.figure()
plt.plot(Centers_x, Centers_y, '.')
plt.title(str(MaxStep) + ' iterations with Decaying Learning Rate')
plt.xlim([-1, 1])
plt.ylim([-1, 1])

plt.show()