from aspyre.orientation import Orient3D
import logging

logger = logging.getLogger(__name__)


class CommLineEV(Orient3D):
    """
    Define a derived class to estimate 3D orientations using eigenvector method described as below:
    A. Singer and Y. Shkolnisky, Three-Dimensional Structure Determination from Common Lines in Cryo-EM by
    Eigenvectors and Semidefinite Programming, SIAM J. Imaging Sciences, 4, 543-572 (2011).

    """

    def __init__(self, src):
        """
        constructor of an object for estimating 3D orientations
        """
        pass

    def estimate(self):
        """
        perform estimation of orientations
        """
        pass

    def output(self):
        """
        Output the 3D orientations
        """
        pass




