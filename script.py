from functions import *
import numpy as np

if __name__ == "__main__":

    pX = [1/4] * 4
    pX = np.array(pX)
    pY = [1/2, 1/4, 1/8, 1/8]
    pY = np.array(pY)

    pXYSZ = drawSamples(1000, pX, pY)

    # Marginals and entropy of one random variable
    pX = marginalize(pXYSZ, (1, 2, 3))
    pY = marginalize(pXYSZ, (0, 2, 3))
    pS = marginalize(pXYSZ, (0, 1, 3))
    pZ = marginalize(pXYSZ, (0, 1, 2))

    #print("P(X) : ", pX)
    #print("P(Y) : ", pY)
    #print("P(S) : ", pS)
    #print("P(Z) : ", pZ)

    ## Q2

    # Compute the entropies of one random variable
    entX = entropy(pX)
    entY = entropy(pY)
    entS = entropy(pS)
    entZ = entropy(pZ)
    print("\nEntropies:\n")
    print("H(X) = ", entX)
    print("H(Y) = ", entY)
    print("H(S) = ", entS)
    print("H(Z) = ", entZ)

    ## Q3 

    # Joint probabilities and entropies
    pXY = marginalize(pXYSZ, (2, 3))
    pXS = marginalize(pXYSZ, (1, 3))
    pYZ = marginalize(pXYSZ, (0, 2))
    pSZ = marginalize(pXYSZ, (0, 1))

    entXY = joint_entropy(pXY)
    entXS = joint_entropy(pXS)
    entYZ = joint_entropy(pYZ)
    entSZ = joint_entropy(pSZ)

    print("\nJoint entropies:\n")
    print("H(X, Y) = ", entXY)
    print("H(X, S) = ", entXS)
    print("H(Y, Z) = ", entYZ)
    print("H(S, Z) = ", entSZ)

    ## Q4

    # Conditional probabilities
    pZX = np.transpose(marginalize(pXYSZ, (1,2)))
    pSX = np.transpose(pXS)

    print("\nConditional entropies:\n")
    print("H(X|Y) = ", conditional_entropy(pXY))
    print("H(Z|X) = ", conditional_entropy(pZX))
    print("H(S|X) = ", conditional_entropy(pSX))
    print("H(S|Z) = ", conditional_entropy(pSZ))

    ## Q5

    # Conditional joint entropies
    pXYS = marginalize(pXYSZ, (3))
    pSYX = np.transpose(pXYS, axes=(2,1,0))

    print("\nConditional joint entropies:\n")
    print("H(X,Y|S) = ", cond_joint_entropy(pXYS))
    print("H(S,Y|X) = ", cond_joint_entropy(pSYX))

    ## Q6

    # Mutual informations
    print("\nMutual information:\n")
    print("I(X;Y) = ", mutual_information(pXY))
    print("I(X;S) = ", mutual_information(pXS))
    print("I(Y;Z) = ", mutual_information(pYZ))
    print("I(S;Z) = ", mutual_information(pSZ))


    ## Q7

    # Conditional information
    print("\nConditional mutual information:\n")
    print("I(X;Y|S) = ", cond_mutual_information(pXYS))
    print("I(S;Y|X) = ", cond_mutual_information(pSYX))