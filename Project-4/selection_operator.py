import matplotlib.pyplot as plt
import numpy as np


class GASettings:
    def __init__(self, numParents=10, numChildren=50):
        self.numParents = numParents;
        self.numChildren = numChildren;
        self.numGenerations = 5;
        self.selection = 'truncation';
        self.slope = 15;  # for linear ranking selection
        self.tournamentSize = 2;
        self.crossover = 'uniform';
        self.mutation = 0.05;  # std of normal distribution computed as mutation*value
        self.eliteCount = 1;


def select_pool(G, Gcost, settings):
    """ Select a breding pool from the previous generation.
    G is numChildren x numParameters
    Gcost is numChildren long and is sorted in ascending order.
    Returns (P,Pcost)
    where
    P is numParents x numParameters
    Pcost is numParents long.
    """
    if Gcost.size != settings.numChildren:
        print("problem")
        return
    P = np.zeros((settings.numParents, G.shape[1]))
    Pcost = np.zeros((settings.numParents,))

    if settings.selection == 'uniform':
        for i in range(settings.numParents):
            idx = np.random.choice(G.shape[0])
            P[i, :] = G[idx, :]
            Pcost[i] = Gcost[idx]

    if settings.selection == 'truncation':
        for i in range(settings.numParents):
            P[i, :] = G[idx, :]
            Pcost[i] = Gcost[idx]

    elif settings.selection == 'tournament':
        for i in range(settings.numParents):
            idxs = np.random.choice(G.shape[0], (settings.tournamentSize,))
            # pick the best one, which is smaller index (since costs sorted)
            idx = idxs.min()
            P[i, :] = G[idx, :]
            Pcost[i] = Gcost[idx]


    if settings.selection == 'linear_ranking':
        for i in range(settings.numParents):
            pass




    else:
        print("Unknown selection operator: ", settings.selection)
    return (P, Pcost)




def selection_test():
    # Test the selection operators
    # Make fake parameters, using the index as the values (so we can differentiate them)
    lam = 10000  # number of individuals in generation
    mu = 1000  # number of individuals in breeding pool
    NP = 5  # num of parameters per individual
    G = np.zeros((lam, NP))
    for i in range(lam):
        G[i, :] = i
    Gcost = np.linspace(0.1, 500, lam)
    settings = GASettings(numParents=mu, numChildren=lam)

    settings.selection = 'uniform'
    (P, Pcost) = select_pool(G, Gcost, settings)
    plt.figure(figsize=(8, 4))
    plt.hist(Pcost);
    plt.title("Breeding Pool Chosen Uniformly")
    plt.xlabel('Cost')
    plt.ylabel('Frequency')
    plt.show()

if __name__ == '__main__':
    selection_test()