import argparse

from aspyre import config
from aspyre.source.star import Starfile
from aspyre.basis.fb_3d import FBBasis3D
from aspyre.estimation.mean import MeanEstimator
from aspyre.estimation.covar import CovarianceEstimator
from aspyre.estimation.noise import WhiteNoiseEstimator


def parse_args():
    parser = argparse.ArgumentParser(description='Run Covariance Estimator on a Starfile source.')
    parser.add_argument('--starfile', required=True)
    parser.add_argument('--pixel_size', default=1, type=float)
    parser.add_argument('--ignore_missing_files', action='store_true')
    parser.add_argument('--max_rows', default=None, type=int)
    parser.add_argument('-L', default=16, type=int)
    parser.add_argument('--cg_tol', default=None, type=float)

    return parser.parse_args()


if __name__ == '__main__':
    opts = parse_args()
    source = Starfile(
        opts.starfile,
        pixel_size=opts.pixel_size,
        ignore_missing_files=opts.ignore_missing_files,
        max_rows=opts.max_rows
    )

    L = opts.L
    source.set_max_resolution(L)
    source.cache()

    source.whiten()
    basis = FBBasis3D((L, L, L))
    mean_estimator = MeanEstimator(source, basis, batch_size=8192)
    mean_est = mean_estimator.estimate()

    noise_estimator = WhiteNoiseEstimator(source, batchSize=500)
    # Estimate the noise variance. This is needed for the covariance estimation step below.
    noise_variance = noise_estimator.estimate()
    print(f'Noise Variance = {noise_variance}')

    # Passing in a mean_kernel argument to the following constructor speeds up some calculations
    covar_estimator = CovarianceEstimator(source, basis, mean_kernel=mean_estimator.kernel)

    override_dict = {}
    if opts.cg_tol is not None:
        override_dict['covar.cg_tol'] = opts.cg_tol

    with config.override(override_dict):
        covar_est = covar_estimator.estimate(mean_est, noise_variance)
