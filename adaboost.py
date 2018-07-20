import numpy as np
import torch
from haar_features import HaarFeatureMaker
from threshold_finder import get_best_threshold_func


"""
This module creates classes for the adaboost algorithm.
"""


class Predictor:
    """
    Base class for a "predictor" object, has a simple error method    
    """
    def __init__(self):
        super(Predictor)
    
    def error(self, featmtx, labels, weights=None):
        if weights is None: weights = np.ones((featmtx.shape[0],))/featmtx.shape[0]
        preds = self(featmtx)
        err = weights.dot(preds != labels) / weights.sum()
        return err

class Thresholder(Predictor):

    def __init__(self,haar,ind,threshold,sgn):
        """
        Generates a simple decision stump function.
        Objects of this class can be called as functions.

        Parameters
        ----------
        ind: the index of the feature used for thresholding
        threshold: the threshold for this decision stump
        sign: the sign of the threshold
        
        """
        super(Thresholder)
        self.feat_index = ind
        self.threshold = threshold
        self.sign = sgn
        self.haar = haar
    
    def __call__(self, Xinput):
        signed_val = self.sign * (Xinput[:,self.feat_index] - self.threshold)
        return (signed_val > 0).astype(int)
    
    def get_tensor(self, X):
        mask = self.haar.get_true_mask(self.feat_index)
#         set_trace()
        feat_vals = X.mm(torch.Tensor(mask))
        sigmoid_preds = torch.sigmoid(self.sign*(feat_vals - self.threshold)/feat_vals.std())
        return sigmoid_preds.reshape((-1,)) 

    
class GroupThresholder(Predictor):

    def __init__(self):
        """This class describes the weighted majority vote of decision stumps.
        Used as the primary output of a boosting algorithm.
        The objection self.alphas_and_thresholds is a list of (weight, predictor) pairs
        Initial list is empty until functions are added.        
        """
        super(GroupThresholder)
        self.alphas_and_threshes = []
    
    def add(self, alpha, hypothesis):
        self.alphas_and_threshes.append((alpha, hypothesis))

    def __call__(self, featmtx):
        hweights, hyps = zip(*self.alphas_and_threshes)
        normweights = np.array(hweights)/sum(hweights)
        predsmtx = np.array([predictor(featmtx) for predictor in hyps])
        weightedpreds = normweights.dot(predsmtx)
        return (weightedpreds > 0.5).astype(int)
    
    def get_tensor(self,X):
        
        hweights, thresh_funcs = zip(*self.alphas_and_threshes)
        tensor_weights = torch.Tensor(np.array(hweights).reshape((-1,1)))
        tensor_weights = tensor_weights / torch.sum(tensor_weights)
        pred_list = torch.stack(
            [thresh_func.get_tensor(X) for thresh_func in thresh_funcs],
            dim = 1
        )
#         return pred_list, tensor_weights
        combined_preds = pred_list.mm(tensor_weights)
        return combined_preds.reshape((-1,))

        
        
    
class Booster:
    def __init__(self,traindata,trainlabels,haars,testdata=None,testlabels=None):
        """Generates a boosting algorithm state. 
        Parameters
        ----------
        traindata: training data, N x M, N examples and M haar features
        trainlabels: testing data, vector of size N, of 0/1 labels
        """
        super(Booster)
        self.haars = haars
        self.traindata = self.haars.get_haar_feats(traindata)
        self.numdat, self.numfeat = self.traindata.shape
        self.trainlabels = trainlabels
        if testdata is None:
            self.testdata = None
        else:
            self.testdata = self.haars.get_haar_feats(testdata)
        self.testlabels = testlabels

        # The main parameters of a boosting algorithm
        # We keep the weights in log space for numerical stability
        self._logweights = np.zeros((self.numdat,))

        # The weighted list of decision stumps is stored here:
        self._group_hypothesis = GroupThresholder()
        
    def weights(self):
        # Returns the weights, which are stored in log form
        wts = np.exp(self._logweights)
        return wts / np.sum(wts)
    
    def predict_weighted_majority(self,featmtx):
        return self._group_hypothesis(featmtx)

    def train_error(self):
        return self._group_hypothesis.error(self.traindata, self.trainlabels)

    def test_error(self):
        if self.testdata is None: return np.inf
        else: return self._group_hypothesis.error(self.testdata, self.testlabels)

    def boost_iter(self):
        # This is the primary iteration of the boosting algorithm

        # Weak learner is generated here
        thresh_func_params, perf = get_best_threshold_func(
            self.traindata, self.trainlabels, self.weights())
        thresh_func = Thresholder(self.haars, *thresh_func_params)

        # Predictions of weak learner, and error rate are calculated
        preds = thresh_func(self.traindata)
        err = thresh_func.error(self.traindata,self.trainlabels,self.weights())
        print("train error of thresh_func %.4f" % err)

        # These values are needed by boosting
        beta = (err + 0.001)/(1-err - 0.001)
        alpha = -np.log(beta)
        corrects = (preds == self.trainlabels)

        # Calculate the scores on examples, used for reweighting
        scores = beta ** corrects

        # Update the log weights
        self._logweights += np.log(scores)

        # Store the weighted predictor in the group thresholder object
        self._group_hypothesis.add(alpha, thresh_func)
    
    # def show_some_images(self):
    #     wrong_indices = testdata_df.iloc[self.predict_weighted_majority(self.testdata) != self.testlabels].index
    #     wrongers = X[wrong_indices]
    #     wrongerss = wrongers[np.random.randint(0,wrongers.shape[0],size=5),:,:]
    #     for ind in range(5):
    #         img = wrongerss[ind].reshape([28,28])
    #         plt.imsave('incorrect_%d.png' % ind,img)
    
    def train(self, numiter=10):
        for ind in range(numiter):
            self.boost_iter()
#             self.show_some_images()
            print("boost test/train error is %.4f, %.4f" % (self.test_error(), self.train_error()))



if __name__ == '__main__':
    import mnist_loader

    X_train, Y_train, X_test, Y_test = mnist_loader.get_mnist_9_4()
    
    haar = HaarFeatureMaker(X_train.shape[1])

    boost = Booster(X_train, Y_train, haar, X_test, Y_test)
    boost.train(numiter=5)


