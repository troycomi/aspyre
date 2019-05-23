import numpy as np
from unittest import TestCase
from aspyre.cov3d.estimation.kernel import FourierKernel

import os.path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class FourierKernelTestCase(TestCase):
    def setUp(self):
        self.kernel = FourierKernel(np.load(os.path.join(DATA_DIR, 'fourier_kernel_16_16_16.npy')), centered=False)

    def tearDown(self):
        pass

    def test1(self):
        toeplitz = self.kernel.toeplitz()
        self.assertEqual(toeplitz.shape, (8, 8, 8, 8, 8, 8))
        self.assertTrue(
            np.allclose(
                toeplitz[:, :, 0, 0, 0, 2],
                np.array([
                    [-3.99237964e-04, -5.58560540e-04, -4.61112126e-04, -1.33411959e-05, 1.80705421e-04, 5.17746266e-05, -1.11463014e-04, -6.67081913e-05],
                    [-5.73768339e-04, -5.82092151e-04, -3.57612298e-04, 6.28258349e-05, 2.10987753e-04, -1.37316420e-05, -1.81071970e-04, -1.65530946e-05],
                    [-4.50893916e-04, -3.94886069e-04, -1.31166336e-04, 1.83019380e-04, 1.86107689e-04, -8.73301760e-05, -1.89300568e-04, 4.55726404e-05],
                    [-7.88352190e-05, -5.64255715e-05, 1.79488547e-04, 2.42688577e-04, 7.89244223e-05, -1.95445391e-04, -1.64644473e-04, 8.46998155e-05],
                    [1.78478949e-04, 1.10234592e-04, 1.29420368e-04, -9.66549123e-06, -1.59117772e-04, -2.06353434e-04, 8.77526109e-05, 2.81196553e-04],
                    [-1.48982726e-05, -5.52274614e-05, -9.01011008e-05, -1.95092929e-04, -2.12131679e-04, -2.77411455e-05, 1.63609919e-04, 1.70907035e-04],
                    [-1.23197358e-04, -2.01485047e-04, -2.22623392e-04, -1.35038150e-04, 2.55815121e-05, 1.63564575e-04, 7.31561740e-05, -2.58645450e-05],
                    [-1.00527395e-04, -5.97040562e-05, 6.75393894e-05, 2.12094223e-04, 2.47279182e-04, 1.78254995e-04, -3.36604789e-05, -2.02765747e-04]
                ])
            )
        )