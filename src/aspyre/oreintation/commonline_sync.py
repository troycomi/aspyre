from aspyre.oreintation import Oreint3D
import logging

logger = logging.getLogger(__name__)


class CommLineSync(Oreint3D):
    """
    Define a class to estimate 3D oreintations using Synchronization described as below:
    Y. Shkolnisky, and A. Singer, Viewing Direction Estimation in Cryo-EM Using Synchronization,
    SIAM J. Imaging Sciences, 5, 1088-1110 (2012).

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




