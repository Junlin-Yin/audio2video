import numpy as np

def PCA(data):
    '''
    PCA	Principal Component Analysis

    Input:
      data      - Data numpy array. Each row vector of fea is a data point.
    Output:
      eigvector - Each column is an embedding function, for a new
                  data point (row vector) x,  y = x*eigvector
                  will be the embedding result of x.
      eigvalue  - The sorted eigvalue of PCA eigen-problem.
    '''

    # YOUR CODE HERE
    # Hint: you may need to normalize the data before applying PCA
    # begin answer
    N, P = data.shape
    mu = np.mean(data, axis=0)
    X = data - mu             # mu(X) = 0
    S = np.matmul(X.T, X)/N   # S.shape = (P, P) 
    ex, ev = np.linalg.eig(S)
    tmp = zip(ex, range(len(ex)))
    stmp = sorted(tmp, key=lambda x:x[0], reverse=True)
    eigvalue = np.array([x[0] for x in stmp])
    eigvector = np.vstack([ev[:,x[1]] for x in stmp]).T
    return eigvector, eigvalue
    # end answer