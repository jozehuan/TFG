from abc import ABC, abstractmethod

class Explanation(ABC):
    """
    Abstract base class for a explanation

    Attr
    ----------
        _model : flex.model.FlexModel
            The node's FlexModel
        _explainer 
            The explainer instance
        _id_data : int
            data ID
        _explainer_kwargs : dict
            dictionaire for explainer parameters
        _label 
            data label
    """

    def __init__(self, model, exp, id_data, label, *args, **kwargs):
        self._model = model
        self._explainer = exp
        self._id_data = id_data
        self._explain_kwargs = kwargs
        self._label = label

    @abstractmethod
    def get_explanation(self, data, label):
        """
        Get computed explanations

        Attr
        ----------
            data : torch.Tensor
                explained data 
            label 
                data label
        """
        pass