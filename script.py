from functions import *
import numpy as np

if __name__ == "__main__":

    pX = [1/4] * 4
    pX = np.array(pX)
    pY = [1/2, 1/4, 1/8, 1/8]
    pY = np.array(pY)

    pXYSZ = drawSamples(1000000, pX, pY)

    # Marginals and entropy of one random variable
    pX = marginalize(pXYSZ, (1, 2, 3))
    pY = marginalize(pXYSZ, (0, 2, 3))
    pS = marginalize(pXYSZ, (0, 1, 3))
    pZ = marginalize(pXYSZ, (0, 1, 2))

    print("P(X) : ", pX)
    print("P(Y) : ", pY)
    print("P(S) : ", pS)
    print("P(Z) : ", pZ)

    entX = entropy(pX)
    entY = entropy(pY)
    entS = entropy(pS)
    entZ = entropy(pZ)

    print("H(X) : ", entX)
    print("H(Y) : ", entY)
    print("H(S) : ", entS)
    print("H(Z) : ", entZ)

    # Joint probabilities and entropies
    pXY = marginalize(pXYSZ, (2, 3))

    print("P(X,Y) : \n", pXY)

    pXS = marginalize(pXYSZ, (1, 3))
    pYZ = marginalize(pXYSZ, (0, 2))
    pSZ = marginalize(pXYSZ, (0, 1))

    entXY = joint_entropy(pXY)
    entXS = joint_entropy(pXS)
    entYZ = joint_entropy(pYZ)
    entSZ = joint_entropy(pSZ)

    print("H(X, Y) : ", entXY)
    print("H(X, S) : ", entXS)
    print("H(Y, Z) : ", entYZ)
    print("H(S, Z) : ", entSZ)

    print(conditional_entropy(pXY))

    print(mutual_information(pXY))
    print(mutual_information(pXS))

    pXYS = marginalize(pXYSZ, (3))
    print(cond_joint_entropy(pXYS))