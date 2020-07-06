import random as rand
import math

def addList(W1, W2, Sub):  # pairwise sum/sub of two lists
    if not W1:
        return W2
    elif not W2:
        return W1
    elif type(W1) == list:
        return [addList(a, b, Sub) for a, b in zip(W1, W2)]
    else:
        if not Sub:
            return W1 + W2
        else:
            return W1 - W2

def readNums(File): # to read numbers only from a file for weights and patterns initialization
    f = open(File, 'r')
    data = []
    for l in f:
        l = l[0:l.find('#')] # anything followed by a '#' will be ignored
        data = data + [float(x) for x in l.split()] # two numbers must be seprated with a whitespace
    f.close()
    return data

def readPatterns(File, N): # to read training patterns from a file using readNums function and form them in a desirable way for further process
    data = readNums(File)
    P = int(len(data)/N)
    return [[data[p * N + n] for n in range(N)] for p in range(P)]
    # patterns array, contains P lists consist of coordination for each point in the N-dim input space

def MultiNeuralGas(M, N, K, Z0, Zend, Width, PartnerSizes, TrainingPatterns, MaxStep, RandomSeed):
    ### Initialization ###
    rand.seed(RandomSeed)
    if (len(PartnerSizes) != M) or (sum(PartnerSizes) != K):
        PartnerSizes = [int(K / M) for m in range(M)]
        for i in range(K % M):
            PartnerSizes[i] = PartnerSizes[i] + 1
    NeuronGroupInd = [] #A list that shows each neuron's group index
    for m in range(M):
        NeuronGroupInd = NeuronGroupInd + [m for k in range(PartnerSizes[m])]
    Centers = [[rand.random() * 2 - 1 for n in range(N)] for k in range(K)] #Equaly distributed random points within unit cube
    if type(TrainingPatterns) == str: #Training patterns could either read from a file or pass directly as a list
        TrainingPatterns = readPatterns(TrainingPatterns, N)
    ### Learning ###
    def EqDis(A, B):  # return the euclidean distance between A and B
        return sum([(a - b) ** 2 for a, b in zip(A, B)])
    def LearningRule(time):
        return (math.exp((MaxStep - 1 - time)/100)/math.exp((MaxStep - 1)/100)) * (Z0 - Zend) + Zend
    def h(Dis):
        return math.exp(-0.5 * (Dis ** 2) / (Width ** 2))
    for t in range(MaxStep):
        rand.shuffle(TrainingPatterns) #changing patterns order after using every pattern
        Zeta = LearningRule(t)
        for Pattern in TrainingPatterns:
            Distances = [EqDis(Pattern, Center) for Center in Centers]
            WinnerPointIndex = Distances.index(min(Distances))
            WinnerGroupIndex = NeuronGroupInd[WinnerPointIndex]
            Begin = sum(PartnerSizes[:WinnerGroupIndex])
            End = Begin + PartnerSizes[WinnerGroupIndex]
            SortedListL = [(Distances[i], i) for i in range(Begin, End)]
            SortedListL.sort()
            SortedListL = [x[1] for x in SortedListL]
            for k in range(Begin, End):
                Coeff = Zeta * h(SortedListL.index(k))
                dCenter = [Coeff * (b - a) for a, b in zip(Centers[k], Pattern)] # Learning rule
                Centers[k] = addList(Centers[k], dCenter, 0)
    return Centers