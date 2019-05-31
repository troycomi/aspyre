import logging

logger = logging.getLogger(__name__)


class CommLineGCAR(Oreint3D):
    """
    Define a class to estimate 3D oreintations using Globally Consistent Angular Reconstitution described as below:
    R. Coifman, Y. Shkolnisky, F. J. Sigworth, and A. Singer, Reference Free Structure Determination through
    Eigenvestors of Center of Mass Operators, Applied and Computational Harmonic Analysis, 28, 296-312 (2010).

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




