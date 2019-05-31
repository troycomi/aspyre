import logging

logger = logging.getLogger(__name__)


class CommLineEV(Oreint3D):
    """
    Define a class to estimate 3D oreintations using eigenvectors described as below:
    A. Singer and Y. Shkolnisky, Three-Dimensional Structure Determination from Common Lines in Cryo-EM by
    Eigenvectors and Semidefinite Programming, SIAM J. Imaging Sciences, 4, 543-572 (2011).

    """

    def __init__(self, src):
        """
        constructor of an object for estimating 3D oreintations
        """
        pass

    def estimate(self):
        """
        perform classifying 2D images
        """
        pass

    def output(self):
        """
        Output the clean images
        """
        pass




