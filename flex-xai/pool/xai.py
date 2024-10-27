
import numpy as np
import warnings

from lime.wrappers.scikit_image import SegmentationAlgorithm

from skimage.color import gray2rgb

class Image_SubmodularPick(object):
    """Class for submodular pick for images

    Saves a representative sample of explanation objects using SP-LIME,
    as well as saving all generated explanations

    First, a collection of candidate explanations are generated
    (see explain_instance). From these candidates, num_exps_desired are
    chosen using submodular pick. (see marcotcr et al paper)."""

    def __init__(self,
                 explainer,
                 data,
                 predict_fn,
                 method='sample',
                 sample_size=1000,
                 num_exps_desired=10,
                 **kwargs):

        """
        Args:
            data: a numpy array where each row is a single input into predict_fn
            predict_fn: prediction function. For classifiers, this should be a
                    function that takes a numpy array and outputs prediction
                    probabilities. For regressors, this takes a numpy array and
                    returns the predictions. For ScikitClassifiers, this is
                    `classifier.predict_proba()`. For ScikitRegressors, this
                    is `regressor.predict()`. The prediction function needs to work
                    on multiple feature vectors (the vectors randomly perturbed
                    from the data_row).
            method: The method to use to generate candidate explanations
                    method == 'sample' will sample the data uniformly at
                    random. The sample size is given by sample_size. Otherwise
                    if method == 'full' then explanations will be generated for the
                    entire data. l
            sample_size: The number of instances to explain if method == 'sample'
            num_exps_desired: The number of explanation objects returned


        Sets value:
            sp_explanations: A list of explanation objects that has a high coverage
            explanations: All the candidate explanations saved for potential future use.
              """

        # Parse args
        if method == 'sample':
            if sample_size > len(data):
                warnings.warn("""Requested sample size larger than
                              size of input data. Using all data""")
                sample_size = len(data)
            all_indices = np.arange(len(data))
            np.random.shuffle(all_indices)
            sample_indices = all_indices[:sample_size]
        elif method == 'full':
            sample_indices = np.arange(len(data))
        else:
            raise ValueError('Method must be \'sample\' or \'full\'')

        # Generate Explanations
        n_pixels = data[0].size(1) * data[0].size(2)
        segmenter = SegmentationAlgorithm('slic', n_segments=n_pixels)
        
        self.explanations = []
        for i in sample_indices:
            img_rgb = gray2rgb(data[i])
            self.explanations.append(
                explainer.explain_instance(img_rgb, classifier_fn = predict_fn, segmentation_fn=segmenter, **kwargs) )
                
        # Error handling
        try:
            num_exps_desired = int(num_exps_desired)
        except TypeError:
            return("Requested number of explanations should be an integer")
        if num_exps_desired > len(self.explanations):
            warnings.warn("""Requested number of explanations larger than
                           total number of explanations, returning all
                           explanations instead.""")
        num_exps_desired = min(num_exps_desired, len(self.explanations))

        # Find all the explanation model features used. Defines the dimension d'
        # In this case, all the pixels are used, so d' = lenght * width of the image
        d_prime = n_pixels  


        # Create the n x d' dimensional 'explanation matrix', W
        W = np.zeros((len(self.explanations), d_prime))
        for i, exp in enumerate(self.explanations):
            for label_weights in exp.local_exp.values():
                for feature, value in label_weights:
                    W[i, feature] += value

        # Create the global importance vector, I_j described in the paper
        importance = np.sum(abs(W), axis=0)**.5

        # Now run the SP-LIME greedy algorithm
        remaining_indices = set(range(len(self.explanations)))
        V = []
        for _ in range(num_exps_desired):
            best = 0
            best_ind = None
            current = 0
            for i in remaining_indices:
                current = np.dot(
                        (np.sum(abs(W)[V + [i]], axis=0) > 0), importance
                        )  # coverage function
                if current >= best:
                    best = current
                    best_ind = i
            V.append(best_ind)
            remaining_indices -= {best_ind}

        self.sp_explanations = [self.explanations[i] for i in V]
        self.V = V
