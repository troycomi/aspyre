from aspyre.oreintation import Oreint3D
import logging

logger = logging.getLogger(__name__)


class CommLineLUD(Oreint3D):
    """
    Define a class to estimate 3D oreintations using Least Unsquared Deviations described as below:
    L. Wang, A. Singer, and  Z. Wen, Orientation Determination of Cryo-EM Images Using Least Unsquared Deviations,
    SIAM J. Imaging Sciences, 6, 2450-2483 (2013).

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




