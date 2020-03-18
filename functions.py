import math
import numpy as np


def drawSamples(nbSamples, pX, pY):
    """Draws 2*nbSamples following the probability distributions pX and pY and returns
    the joint probability distribution of X,Y,Z,S approximated using the samples.   
        
    
    Arguments:
        nbSamples {int} -- Number of samples to draw from each marginal probability (X and Y) to estimate the joint one
        pX {array of float} -- Probability distribution of X
        pY {array of float} -- Probability distribution of Y
    """

    samplesX = np.random.choice(4, nbSamples, p=pX)
    samplesY = np.random.choice(4, nbSamples, p=pY)

    jointProba = np.zeros((4, 4, 7, 2))

    for n in range(nbSamples):
        jointProba[samplesX[n]][samplesY[n]][samplesX[n]+samplesY[n]][1 if samplesX[n] == samplesY[n] else 0] += 1

    return jointProba / nbSamples

def marginalize(jointProba, axis):
    """Returns the marginalized probability distribution of jointProba along the axis defined in axis
    
    Arguments:
        jointProba {list} -- Joint probability to marginalize.
        axis {tuple of int} -- Axis along which to marginalize.

    Returns:
        list -- Marginalized list.
    """
    return np.sum(jointProba, axis)

def entropy(pd):
    """Computes the entropy of the discrete random variable whose probability distribution is given in pd.
    
    Arguments:
        pd {array of float} -- Array representing the probability distribution of a random variable.
    
    Returns:
        float -- The entropy computed via the probabilities provided in pd.
    """
    entp = 0
    for proba in pd:
        if proba == 0:
            continue
        entp -= proba*math.log(proba, 2)
    return entp

def joint_entropy(pd):
    """Computes the entropy of the pd.ndim discrete random variables whose joint probability distribution is given in pd.
    
    Arguments:
        pd {multidim array of float} -- Array representing a joint probability distribution of discrete random variables.
    
    Returns:
         float -- The entropy computed via the probabilities provided in pd.
    """
    # Attention to multidimension and proba equal to 0 which makes a matherror for log.
    return entropy(pd.flatten())

def conditional_entropy(pdXY):
    # Computes the entropy of X conditioned on Y.
    # Lines : values of X. Columns : values of Y.
    pdY = marginalize(pdXY, (0))

    return joint_entropy(pdXY) - entropy(pdY)
    """
    tester avec probas ainsi.
    pdcond = pdXY / pdY

    return entropy(pdcond.flatten())
    """

def mutual_information(pdXY):

    pdX = marginalize(pdXY, (1))
    return entropy(pdX) - conditional_entropy(pdXY)

def cond_joint_entropy(pdXYZ):

    pdZ = marginalize(pdXYZ, (0, 1))

    return joint_entropy(pdXYZ) - entropy(pdZ)

def cond_mutual_information(pdXYZ):
    pXZ = marginalize(pdXYZ, (1))
    pZ = marginalize(pdXYZ, (0,1))
    pYZ = marginalize(pdXYZ, (0))
    return joint_entropy(pXZ) - 2 * entropy(pZ) - cond_joint_entropy(pdXYZ) + joint_entropy(pYZ)
    