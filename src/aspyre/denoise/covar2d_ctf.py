import logging
import nuddmpy as np
from scipy.sparse.linalg import LinearOperator, cg
from scipy.linalg import norm
from tqdm import tqdm

from aspyre import config
from aspyre.imaging.threed import rotated_grids
from aspyre.nfft import anufft3
from aspyre.utils import mdim_ifftshift, vol_to_vec, vecmat_to_volmat, volmat_to_vecmat, ensure, \
    symmat_to_vec_iso, vec_to_symmat_iso, make_symmat
from aspyre.utils.matlab_compat import m_reshape

from aspyre.utils.blk_diag_func import *
from aspyre.denoise.covar2d import RotCov2D


logger = logging.getLogger(__name__)

class Cov2DCTF(RotCov2D):
    """
    Define a derived class for denoising 2D images using CTF and Wiener Cov2D method
    """

    def get_mean_ctf(self, coeffs=None, ctf_fb, ctf_idx):
        """
        Calculate the mean vector from the expansion coefficient.
        param b_coeffs: A coefficient vector (or an array of coefficient vectors) to be evaluated.
        :return: The mean value vector for all images.
        """
        if coeffs is None:
            raise RuntimeError('The coefficients need to be calculated!')

        b= np.zeros((self.basis.basis_count,1),dtype=self.as_type)

        A = blk_diag_zeros(blk_diag_partition(ctf_fb[0]), dtype=ctf_fb[0][0].dtype)
        for k in np.unique(ctf_idx[:]).T :
            coeff_k = coeffs[:,ctf_idx== k]
            weight = np.size(coeff_k, 1)/np.size(coeffs,1)
            mean_coeff_k = self.get_mean(coeff_k)
            ctf_fb_k = ctf_fb[k]
            ctf_fb_k_t = blk_diag_transpose(ctf_fb_k)
            b = b + weight*blk_diag_apply(ctf_fb_k_t, mean_coeff_k[:])
            A = blk_diag_add(A, blk_diag_mult(weight, blk_diag_mult(ctf_fb_k_t, ctf_fb_k)))

        mean_ceoff = blk_diag_solve(A, b)

        return mean_coeff

    def get_covar_ctf(self, coeffs=None, ctf_fb, ctf_idx, mean_coeff=None, noise_var, covar_est_opt=None):

        if covar_est_opt is None :
            covar_est_opt = {
            'shrinker':'none', 'verbose': 0, 'max_iter': 250, 'rel_tolerance': 1e-12
            }

        block_partition = blk_diag_partition(ctf_fb[0])
        b_coeff = blk_diag_zeros(block_partition, dtype=coeffs.dtype)
        b_noise = blk_diag_zeros(block_partition, dtype=coeffs.dtype)
        A = cell(numel(ctf_fb), 1)
        M = blk_diag_zeros(block_partition,dtype=ctf_fb[0][0])

        for k = unique(ctf_idx[:])'

            coeff_k = coeff[:, ctf_idx == k]
            weight = np.size(coeff_k, 1)/np.size(coeffs, 1)

            ctf_fb_k = ctf_fb[k]
            ctf_fb_k_t = blk_diag_transpose(ctf_fb_k)

            mean_coeff_k = blk_diag_apply(ctf_fb_k, mean_coeff)
            covar_coeff_k = self.get_covar(coeff_k, mean_coeff_k)

            b_coeff = blk_diag_add(b_coeff, blk_diag_mult(ctf_fb_k_t,
                blk_diag_mult(covar_coeff_k, blk_diag_mult(ctf_fb_k, weight))))

            b_noise = blk_diag_add(b_noise, blk_diag_mult(weight,
                blk_diag_mult(ctf_fb_k_t, ctf_fb_k)))

            A[k] = blk_diag_mult(ctf_fb_k_t, blk_diag_mult(ctf_fb_k, np.sqrt(weight)))

            M = blk_diag_add(M, A[k])

        if covar_est_opt['shrinker'] == 'none':
            b = blk_diag_add(b_coeff, blk_diag_mult(-noise_var, b_noise))
        else:
            b = self.shrink_covar_backward(b_coeff, b_noise, np.size(coeffs, 1),
                noise_var, covar_est_opt['shrinker'])

        cg_opt = covar_est_opt

        covar_coeff = blk_diag_zeros(block_partition, dtype=coeffs.dtype)

        for k = 0:b.size()
            A_k = A[k]
            # cellfun(@(blk)(blk{k}), A, 'uniformoutput', false);
            b_k = b[k]

            S = np.invert(M[k])

            cg_opt.preconditioner = @(x)(precond(S, x));

            covar_coeff{k} = conj_grad(@(x)(apply(A_k, x)), b_k(:), cg_opt);

            covar_coeff{k} = reshape(covar_coeff{k}, size(A_k{1}, 1)*ones(1, 2));
        end
    end


    def _shrink(self, covar_b_coeff, noise_variance, method=None):
        """
        Shrink covariance matrix
        :param covar_b_coeff: Outer products of the mean-subtracted images
        :param noise_variance: Noise variance
        :param method: One of None/'frobenius_norm'/'operator_norm'/'soft_threshold'
        :return: Shrunk covariance matrix
        """
        ensure(method in (None, 'frobenius_norm', 'operator_norm', 'soft_threshold'), 'Unsupported shrink method')

        An = self.basis.mat_evaluate_t(self.mean_kernel.toeplitz())
        if method is None:
            covar_b_coeff -= noise_variance * An
        else:
            raise NotImplementedError('Only default shrink method supported.')

        return covar_b_coeff


    def conj_grad(self, b_coeff):

        def precond_fun(covar_coeff):
        return symmat_to_vec_iso(self.apply_kernel(vec_to_symmat_iso(covar_coeff), kernel=self.precond_kernel))

        b_coeff = symmat_to_vec_iso(b_coeff)
        N = b_coeff.shape[0]

        operator = LinearOperator((N, N), matvec=kernel_fun)
        M = None if self.precond_kernel is None else LinearOperator((N, N), matvec=precond_fun)

        tol = config.covar.cg_tol
        target_residual = tol * norm(b_coeff)

        def cb(xk):
            logger.info(f'Delta {norm(b_coeff - kernel_fun(xk))} (target {target_residual})')

            x, info = cg(operator, b_coeff, M=M, callback=cb, tol=tol)

        if info != 0:
            raise RuntimeError('Unable to converge!')
        return vec_to_symmat_iso(x)

    def precond_fun(S, x):
        p=np.size(S, 0)
        x=m_reshape(x, p*np.ones((1,2)))
        y=S @ x @ S
        y= symmat_to_vec_iso(self.apply_kernel(vec_to_symmat_iso(covar_coeff), kernel=self.precond_kernel))
        retun y

    def get_wiener_ctf(self, coeffs, filter_fb, filter_idx,mean_coeff, covar_coeff, noise_var)


        blk_partition = blk_diag_partition(covar_coeff)
        precision =    class(coeff)

        noise_covar_coeff = blk_diag_mult(noise_var, ...
        blk_diag_eye(blk_partition, precision));

        coeff_est = zeros(size(coeff), precision);

        for k = unique(filter_idx(:))'
            mask = (filter_idx == k);

            coeff_k = coeff(:, mask);

            filter_fb_k = filter_fb{k};
            filter_fb_k_t = blk_diag_transpose(filter_fb_k);

            sig_covar_coeff = ...
            blk_diag_mult(filter_fb_k, ...
            blk_diag_mult(covar_coeff, filter_fb_k_t));

            sig_noise_covar_coeff = blk_diag_add(sig_covar_coeff, ...
            noise_covar_coeff);

            mean_coeff_k = blk_diag_apply(filter_fb_k, mean_coeff);

            coeff_est_k = bsxfun( @ minus, coeff_k, mean_coeff_k);
            coeff_est_k = blk_diag_solve(sig_noise_covar_coeff, coeff_est_k);
            coeff_est_k = blk_diag_apply(...
            blk_diag_mult(covar_coeff, filter_fb_k_t), coeff_est_k);
            coeff_est_k = bsxfun( @ plus, coeff_est_k, mean_coeff);

            coeff_est(:, mask) = coeff_est_k;

        return coeff_est

