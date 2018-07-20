
import numpy as np
from scipy import sparse
from threshold_finder import *

HAAR_MASK_WIDTHS = [1,2,4,8]


class HaarFeatureMaker:
    
    def __init__(self, dim):

        self.mask_widths = HAAR_MASK_WIDTHS
        self.dim = dim

        # To make the tensors from mask indices, we need to store mask params
        self.mask_params = []

        # Records the set of feature masks
        self.feature_mask_matrix = self.make_haar_feat_masks()

        # Flattens the masks
        reshaped_masks = self.feature_mask_matrix.reshape(
            (self.feature_mask_matrix.shape[0],-1))

        # Converts the masks to a sparse format
        self.sparse_reshaped_feat_masks = sparse.csr_matrix(reshaped_masks)

    
    def make_rect_mask(self,i1,i2,j1,j2):
        """Returns a 2d array of a rectangular feature mask AFTER the "cumsumming" operation
        It turns out that the haar masks are much faster to compute on an image IM, when
        we use IM.cumsum(axis=1).cumsum(axis=2). This requires only 4 evaluations per image,
        rather than quadratically many.

        Parameters
        ----------
        i1: top row index
        i2: middle row index
        j1: left column index
        j2: center column index

        Returns
        -------
        out: a (sparse) rectangular mask that is self.dim x self.dim 
        """
        out = np.zeros((self.dim,self.dim))
        if i1 > 0 and j1 > 0: out[i1-1,j1-1] = 1
        out[i2,j2] = 1
        if i1 > 0: out[i1-1,j2] = -1
        if j1 > 0: out[i2,j1-1] = -1
        return out
    
    def make_haar_feature(self,i,j,hsizei,hsizej):
        """A haar feature mask. Ultimately it corresponds to an image 
        matrix of size 2hsizei*2hsizej where:
            top left block is 1s
            top right block is -1s
            bottom left block is -1s
            bottom right block is 1s
        IMPORTANT: this is assuming we are operating on the cumsummed version of the image mtx
        which means that it will ultimately return a sparse matrix with only 9 active coordinates

        Parameters
        ----------
        i: row index of top left corner of haar mask
        j: col index of top left corner of haar mask
        hsizei: num rows of *half* of the haar mask
        hsizej: num columns of *half* of the haar mask
        
        Returns
        -------
        sparse matrix of haar mask operator on cumsummed image mtx

        
        """
        upleft = self.make_rect_mask(i,i+hsizei-1,j,j+hsizej-1)
        upright = self.make_rect_mask(i,i+hsizei-1,j+hsizej,j+2*hsizej-1)
        botleft = self.make_rect_mask(i+hsizei,i+2*hsizei-1,j,j+hsizej-1)
        botright = self.make_rect_mask(i+hsizei,i+2*hsizei-1,j+hsizej,j+2*hsizej-1)
        return upleft + botright - botleft - upright

    def make_haar_feat_masks(self):
        """Generates all haar feature masks across a large range of parameters
        Sweeps through all rows and columns, and all window sizes specified in self.mask_widths
        
        Returns
        -------
        list of all generated haar feature masks
        
        """
        haar_feats = []
        counter = 0
        for i in range(self.dim):
            for j in range(self.dim):
                for hsizei in self.mask_widths:
                    for hsizej in self.mask_widths:
                        if i + 2*hsizei > self.dim or j + 2*hsizej > self.dim: continue
                        self.mask_params.append((i,j,hsizei,hsizej))
                        haar_feats.append(self.make_haar_feature(i,j,hsizei,hsizej))
        return np.array(haar_feats)
    
    def get_true_mask(self,ind):
        """Since masks are stored in a sparse format, this returns the *actual* non-sparse
        mask matrix given the index of the haar feature. Used for generating a tensor.        
        """
        mask = np.zeros((self.dim,self.dim))
        i,j,hsizei,hsizej = self.mask_params[ind]
        mask[i:i+hsizei,j:j+hsizej] = 1
        mask[i+hsizei:i+2*hsizei,j:j+hsizej] = -1
        mask[i:i+hsizei,j+hsizej:j+2*hsizej] = -1
        mask[i+hsizei:i+2*hsizei,j+hsizej:j+2*hsizej] = 1
        return mask.reshape((-1,1))


    def get_haar_feats(self, dataset):
        """Computes haar features for a stack of images.
        Does it efficiently by using the sparse representation of the haar feature masks 

        Parameters
        ----------
        dataset: an N x self.dim x self.dim matrix of images

        Returns
        -------
        haar_feature_mtx: an N x M where M is the number of haar masks generated in make_haar_feat_masks()
        """

        numdat, dim1, dim2 = dataset.shape

        # There's a lot in this line:
        #   First we compute the cumsums of the images along each dimension
        #   Then we reshape these cumsums to flatten
        X_cumsummed = dataset.cumsum(axis=1).cumsum(axis=2).reshape((numdat,-1))

        # Computing haar features now reduces to sparse matrix multiplication
        haar_feature_mtx = self.sparse_reshaped_feat_masks.dot(X_cumsummed.T).T
        return haar_feature_mtx


