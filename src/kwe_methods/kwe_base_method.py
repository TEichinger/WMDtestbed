from abc import ABC, abstractmethod

class BaseKeywordextractor(ABC):
    """Base class for all KWE methods in this project

    Class Variables
    -------
    document: str
        text that is supposed to be used for deriving the Keywords

    sig_size: int
        count of the top k keywords that should be returned from the KWE method

    Notes
    -----
    This is a python abc (abstract base class) and thus cannot be instanced.
    Additional class variables may be added for the implemented KWE methods.
    """

    def __init__(self, document, sig_size):
        self.document = document
        self.sig_size = sig_size

    @abstractmethod
    def run_extraction(self):
        """abstract implementation for the run_extraction function.

           This method is supposed to be the only one that is called by the pipeline code.
           Additional methods may be created but they should only be called from inside of the run_extraction method.

           Returns
           -------
           top_k_keywords : list of str
              list of the top k keywords from the document
           normalized_word_weights : list of floats
              normalized word weights of the top k keywords
        """
        pass
