import numpy as np
from aspyre.denoise import Denoise

import logging
logger = logging.getLogger(__name__)


class RotCov2D(Denoise):
    """
    Define a derived class for denoising 2D images using Cov2D method
    """

    def __init__(self, src, basis, as_type='single'):
        """
          constructor of an object for 2D covariance analysis
        """
        self.basis = basis
        super().__init__(src, as_type)

    def get_mean(self, coeffs=None):
        """
        Calculate the mean vector from the expansion coefficient.
        param coeffs: A coefficient vector (or an array of coefficient vectors) to be evaluated.
        :return: The mean value vector for all images.
        """
        if coeffs is None:
            raise RuntimeError('The coefficients need to be calculated!')

        mask = self.basis._indices["ells"] == 0
        mean_coeff = np.zeros((self.basis.basis_count, 1), dtype=self.as_type)
        mean_coeff = np.mean(coeffs[mask, ...], axis=1)

        return mean_coeff

    def get_covar(self, coeffs=None, mean_coeff=None,  do_refl=false):
        """
        Calculate the covariance matrix from the expansion coefficients.
        param mean_coeff: The mean vector calculated from the `b_coeff`.
        param b_coeffs: A coefficient vector (or an array of coefficient vectors) to be evaluated.
        param do_refl: If true, enforce invariance to reflection (default false).
        :return: The covariance matrix of coefficients for all images.

        """
        if coeffs is None:
            raise RuntimeError('The coefficients need to be calculated!')
        if mean_coeff is None:
            mean_coeff = self.get_mean(coeffs)

        covar_coeff = []
        ind = 0
        ell = 0
        mask =  self.basis._indices["ells"] == ell
        coeff_ell = coeffs[mask,...] - mean_coeff[mask, 0]
        coeffs = coeff_ell@coeff_ell.T/np.size(coeffs, 2)
        covar_coeff.append(coeffs)
        ind +=1

        for ell in range(1, self.basis.ell_max+1):
            mask = self.basis._indices["ells"] == ell
            mask_pos = mask and (self.basis._indices['sgns'] == +1)
            mask_neg = mask and (self.basis._indices['sgns'] == -1)
            covar_ell_diag = (coeffs[mask_pos,:] @ coeff[mask_pos,:].T +
                coeffs[mask_neg,:] @ coeffs[mask_neg,:].T) / (2 * np.size(coeffs, 1))

            if do_refl:
                covar_coeff.append(coval_ell_diag)
                covar_coeff.append(coval_ell_diag)
                ind = ind+2
            else
                covar_ell_off = (coeffs[mask_pos,:] @ coeffs[mask_neg, :].T / np.size(coeffs, 1) - \
                                coeffs[mask_neg,:] @ coeffs[mask_pos, :].T)/(2*np.size(coeffs, 1))

                covar_coeff_blk = np.zeros(2*covar_ell_diag.shape())
                fsize = np.size(covar_coeff_blk, 0)
                hsize = fsize/2
                covar_coeff_blk[0:hsize,0:hsize] = covar_ell_diag[0:hsize,0:hsize]
                covar_coeff_blk[hsize:fsize,hsize:fsize] = covar_ell_diag[0:hsize,0:hsize]
                covar_coeff_blk[0:hsize,hsize:fsize] = covar_ell_off[0:hsize,0:hsize]
                covar_coeff_blk[hsize:fsize,0:hsize] = covar_ell_off.T[0:hsize,0:hsize]
                covar_coeff.append(covar_coeff_blk)
                ind = ind+1

        return covar_coeff


    def get_coeffs(self):
        """
        Apply adjoint mapping to 2D images and obtain the coefficients

        :return: The coefficients of `basis` after the adjoint mapping is applied to the images.
        """
        # TODO It is will be more convenient to calculate the coefficients if they are not done
        raise NotImplementedError('to be implemented')


