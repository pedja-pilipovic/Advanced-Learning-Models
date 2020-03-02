# Libraries
import os
from time import time
import argparse, textwrap
import numpy as np
import pandas as pd
import collections
from itertools import combinations_with_replacement
import cvxopt
cvxopt.solvers.options['show_progress'] = False

# Argument parser
parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-k', required=True, type=int, help='k value for spectrum kernel or kmers')
parser.add_argument('-outputdir', type=str, help='Output directory for Ytek.csv (the directory will be output/ if not provided)', default='output')
parser.add_argument('-inputdir', type=str, help='Input directory for Xtrk.csv, Ytrk.csv and Xtek.csv (the directory will be data/ if not provided)', default='data')
parser.add_argument('-clf', type=str, help=textwrap.dedent('''\
        Classifier:
        - svm (default) [Suport Vector Machine]
        - mnb [Multinomial Naive Bayes on kmers]'''), default='svm')
parser.add_argument('-kernel', type=str, help=textwrap.dedent('''\
        Kernel (for svm classifier only):
        - spectrum (default) [Spectrum kernel]'''), default='spectrum')
args = parser.parse_args()

# Arguments
outputdir = args.outputdir
inputdir = args.inputdir
kernel = args.kernel
classifier = args.clf
k = args.k


## FUNCTION AND CLASS

def get_kmers(seq, k):
    '''
    Get k-mers subsequences of a dna sequence
    '''
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def kmers_counting(seq, k):
    '''
    Get the counting of all possible kmers of a sequence
    '''
    # Creation of the vocabulary for the kmers counting
    x = ''
    for i in range(k):
        x = x + 'actg'
    voc = set([''.join(l) for l in combinations_with_replacement(x,k)])
    
    dic = {}
    for kmer in voc:
        dic[kmer] = 0
    dic.update(collections.Counter(get_kmers(seq, k)))
    return list(collections.OrderedDict(sorted(dic.items())).values())

def classifier_0tom1(y):
    '''
    Get {-1,1} for {0,1}, for svm
    '''
    if (y == 0):
        return int(-1)
    else:
        return 1

def classifier_m1to0(y):
    '''
    Get {0,1} for {-1,1}, for svm
    '''
    if (y == -1):
        return int(0)
    else:
        return 1

    
class svm():
    '''
    Super Vector Machine
    '''
    def __init__(self, lam):
        self.lam = lam
        self.sv_alpha = []
        self.sv = []
        self.sv_y = []
        self.b = []
        
    def fit(self, X, y):
        X = np.array(X)
        m = np.shape(X)[0]
        y = np.array(y)
        # Computing kernel
        K = np.zeros((m, m))
        lam = self.lam
        gamma_val = 1/m
        
        for i in range(m):
            for j in range(m):
                K[i,j] = np.dot(X[i], X[j])
                            
        P = cvxopt.matrix(np.outer(y, y) * K/(2*lam))
        q = cvxopt.matrix(np.ones(m) * -1)
        G = cvxopt.matrix(np.vstack((np.diag(np.ones(m) * -1), np.identity(m))))
        h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m)/m)))

        A = cvxopt.matrix(y.astype('d'), (1,m))
        b = cvxopt.matrix(0.0)
        
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        alpha = np.ravel(sol['x'])
                
        S = np.arange(m)[alpha > 1e-5]
        sv_alpha, sv, sv_y = alpha[S], X[S], y[S]
        
        print(str(len(S)) + ' support vector(s) on ' + str(m))
        
        N = len(sv_alpha)
        f = np.zeros((N))

        for i in range(N):
            f[i] = 1/(2*lam)*np.sum(sv_alpha * sv_y * K[S[i],S])
        
        b = np.mean(sv_y - f)
        
        self.sv_alpha = sv_alpha
        self.sv = sv
        self.sv_y = sv_y
        self.b = b
        self.gamma_val = gamma_val
        
    def predict(self, X):
        np.array(X)
        lam = self.lam
        sv_alpha = self.sv_alpha
        sv = self.sv
        sv_y = self.sv_y
        b = self.b
        
        pred = np.zeros(len(X))
        m = len(X)
        
        K = np.zeros((m, len(sv)))
        
        for i in range(m):
            for j in range(len(sv)):
                K[i,j] = np.dot(X[i], sv[j])
                
        for i in range(len(pred)):
            pred[i] = 1/(2*lam)*np.sum(sv_alpha * sv_y * K[i,:])
            
        return np.sign(pred + b)

    
class mnnaivebayes():
    '''
    Multinomial Naive Bayes Classifier
    '''
    def __init__(self):
        self.logprior = {}
        self.loglikelihood = {}
        self.V = {}
        self.C = {}

    def fit(self, D, C):
        set_C = set(C)
        logprior = {}
        loglikelihood = {}
        bigdoc = {}
        N_doc = len(D)
        V = set(np.concatenate(D))
        for c in set_C:
            N_c = collections.Counter(C)[c]
            logprior[c] = np.log(N_c / N_doc)
            bigdoc[c] = np.concatenate([D[i] for i in range(len(D)) if C[i] == c])
            count_c = collections.Counter(bigdoc[c])
            count_c_sum = np.sum(list(count_c.values()))
            for w in V:
                loglikelihood[(w, c)] = np.log((count_c[w] + 1) / (count_c_sum + len(V)))
        self.logprior = logprior
        self.loglikelihood = loglikelihood
        self.V = V
        self.C = set_C

    def predict(self, doc):
        if len(self.logprior) > 1 and len(self.loglikelihood) > 1 and len(self.V) > 1 and len(self.C) > 1:
            pred_c = []
            for d in doc:
                pmax = -np.inf
                for c in self.C:
                    p = self.logprior[c] + np.sum([self.loglikelihood[(wi, c)] for wi in d if wi in self.V])
                    if p > pmax:
                        pmax = p
                        cmax = c
                pred_c.append(cmax)
            return pred_c
        else:
            print('You need to train before predict')

            
## MAIN

# Creation of the output directory if it not exists
if not os.path.exists(outputdir):
    os.makedirs(outputdir)
    
# Loop on the 3 different files
X_tr = pd.DataFrame()
Y_tr = pd.DataFrame()
for i in [0, 1, 2]:
    print('')
    print('file ' + str(i) + ' ----------')
    # Training and Testing data in pandas dataframe
    X_tr = pd.read_table(str(inputdir)+'/Xtr' + str(i) + '.csv', ',').drop('Id', axis = 1)
    Y_tr = pd.read_table(str(inputdir)+'/Ytr' + str(i) + '.csv', ',').drop('Id', axis = 1)
    X_te = pd.read_table(str(inputdir)+'/Xte' + str(i) + '.csv', ',').drop('Id', axis = 1)
    
    ## Preprocessing
    print('Preprocessing...')
    t = time()
    # All sequences in lower case
    X_tr['seq'] = X_tr.apply(lambda x: x['seq'].lower(), axis = 1)
    X_te['seq'] = X_te.apply(lambda x: x['seq'].lower(), axis = 1)
    
    ## Feature and classifier selection
    if (classifier == 'svm'):
        # Features creation
        X_tr['features'] = X_tr.apply(lambda x: kmers_counting(x['seq'], k), axis = 1)
        X_te['features'] = X_te.apply(lambda x: kmers_counting(x['seq'], k), axis = 1)
        # Labels adaptation
        Y_tr['Class'] = Y_tr.apply(lambda x: classifier_0tom1(x['Bound']), axis = 1)
        # Classifier
        clf = svm(lam=1/(2*len(X_tr['features'])))
    else:
        # Features creation
        X_tr['features'] = X_tr.apply(lambda x: get_kmers(x['seq'], k), axis = 1)
        X_te['features'] = X_te.apply(lambda x: get_kmers(x['seq'], k), axis = 1)
        # Classifier
        clf = mnnaivebayes()
    print('Preprocessing done in ' + str(round(time()-t,3)) + ' s')
    
    # Features and labels adaptation
    X = list(X_tr['features'])
    y = list(Y_tr['Bound'])
    features = list(X_te['features'])
    
    ## Training
    print('Training...')
    t = time()
    clf.fit(X, y)
    print('Training done in ' + str(round(time()-t,3)) + ' s')
    
    ## Prediction
    print('Prediction...')
    t = time()
    labels = clf.predict(features)
    print('Prediction done in ' + str(round(time()-t,3)) + ' s')
    if classifier == 'svm':
        labels = [classifier_m1to0(lbl) for lbl in labels]
    Y_te = pd.DataFrame(labels, columns=['Bound'])
    Y_te.index.name = 'Id'
    
    ## Y_tek file saving
    Y_te.to_csv(outputdir + '/Yte' + str(i) + '.csv')
    print('Prediction file Yte' + str(i) + '.csv created in output/')
    
# Concatenate all testing datasets Y_tek
Y_te = pd.DataFrame()
for i in [0, 1, 2]:
    Y_te = Y_te.append(pd.read_table(outputdir + '/Yte' + str(i) + '.csv', ',').drop('Id', axis = 1), ignore_index = True)

## Y_te file creation
Y_te.index.name = 'Id'
Y_te.to_csv('Yte.csv')
print('')
print('Whole prediction file Yte.csv created')
print('----------------')