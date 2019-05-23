import numpy as np
from unittest import TestCase

from aspyre.source import SourceFilter
from aspyre.imaging.filters import RadialCTFFilter

import os.path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data')


class SimTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testRadialCTFFilter(self):
        filter = RadialCTFFilter(defocus=2.5e4)
        result = filter.evaluate_grid(8)

        self.assertEqual(result.shape, (8, 8))
        self.assertTrue(np.allclose(
            result,
            np.array([
                [ 0.461755701877834,  -0.995184514498978,   0.063120922443392,   0.833250206225063,   0.961464660252150,   0.833250206225063,   0.063120922443392,  -0.995184514498978],
                [-0.995184514498978,   0.626977423649552,   0.799934516166400,   0.004814348317439,  -0.298096205735759,   0.004814348317439,   0.799934516166400,   0.626977423649552],
                [ 0.063120922443392,   0.799934516166400,  -0.573061561512667,  -0.999286510416273,  -0.963805291282899,  -0.999286510416273,  -0.573061561512667,   0.799934516166400],
                [ 0.833250206225063,   0.004814348317439,  -0.999286510416273,  -0.633095739808868,  -0.368890743119366,  -0.633095739808868,  -0.999286510416273,   0.004814348317439],
                [ 0.961464660252150,  -0.298096205735759,  -0.963805291282899,  -0.368890743119366,  -0.070000000000000,  -0.368890743119366,  -0.963805291282899,  -0.298096205735759],
                [ 0.833250206225063,   0.004814348317439,  -0.999286510416273,  -0.633095739808868,  -0.368890743119366,  -0.633095739808868,  -0.999286510416273,   0.004814348317439],
                [ 0.063120922443392,   0.799934516166400,  -0.573061561512667,  -0.999286510416273,  -0.963805291282899,  -0.999286510416273,  -0.573061561512667,   0.799934516166400],
                [-0.995184514498978,   0.626977423649552,   0.799934516166400,   0.004814348317439,  -0.298096205735759,   0.004814348317439,   0.799934516166400,   0.626977423649552]
            ])
        ))

    def testRadialCTFFilterMultiplier(self):
        filter = RadialCTFFilter(defocus=2.5e4) * RadialCTFFilter(defocus=2.5e4)
        result = filter.evaluate_grid(8)

        self.assertEqual(result.shape, (8, 8))
        self.assertTrue(np.allclose(
            result,
            np.array([
                [ 0.461755701877834,  -0.995184514498978,   0.063120922443392,   0.833250206225063,   0.961464660252150,   0.833250206225063,   0.063120922443392,  -0.995184514498978],
                [-0.995184514498978,   0.626977423649552,   0.799934516166400,   0.004814348317439,  -0.298096205735759,   0.004814348317439,   0.799934516166400,   0.626977423649552],
                [ 0.063120922443392,   0.799934516166400,  -0.573061561512667,  -0.999286510416273,  -0.963805291282899,  -0.999286510416273,  -0.573061561512667,   0.799934516166400],
                [ 0.833250206225063,   0.004814348317439,  -0.999286510416273,  -0.633095739808868,  -0.368890743119366,  -0.633095739808868,  -0.999286510416273,   0.004814348317439],
                [ 0.961464660252150,  -0.298096205735759,  -0.963805291282899,  -0.368890743119366,  -0.070000000000000,  -0.368890743119366,  -0.963805291282899,  -0.298096205735759],
                [ 0.833250206225063,   0.004814348317439,  -0.999286510416273,  -0.633095739808868,  -0.368890743119366,  -0.633095739808868,  -0.999286510416273,   0.004814348317439],
                [ 0.063120922443392,   0.799934516166400,  -0.573061561512667,  -0.999286510416273,  -0.963805291282899,  -0.999286510416273,  -0.573061561512667,   0.799934516166400],
                [-0.995184514498978,   0.626977423649552,   0.799934516166400,   0.004814348317439,  -0.298096205735759,   0.004814348317439,   0.799934516166400,   0.626977423649552]
            ])**2
        ))

    def testRadialCTFSourceFilter(self):
        source_filter = SourceFilter(
            [RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)],
            n=42
        )
        result = source_filter.evaluate_grid(8)

        self.assertEqual(result.shape, (8, 8, 7))
        # Just check the value of the last filter for now
        self.assertTrue(np.allclose(
            result[:, :, -1],
            np.array([
                [ 0.461755701877834,  -0.995184514498978,   0.063120922443392,   0.833250206225063,   0.961464660252150,   0.833250206225063,   0.063120922443392,  -0.995184514498978],
                [-0.995184514498978,   0.626977423649552,   0.799934516166400,   0.004814348317439,  -0.298096205735759,   0.004814348317439,   0.799934516166400,   0.626977423649552],
                [ 0.063120922443392,   0.799934516166400,  -0.573061561512667,  -0.999286510416273,  -0.963805291282899,  -0.999286510416273,  -0.573061561512667,   0.799934516166400],
                [ 0.833250206225063,   0.004814348317439,  -0.999286510416273,  -0.633095739808868,  -0.368890743119366,  -0.633095739808868,  -0.999286510416273,   0.004814348317439],
                [ 0.961464660252150,  -0.298096205735759,  -0.963805291282899,  -0.368890743119366,  -0.070000000000000,  -0.368890743119366,  -0.963805291282899,  -0.298096205735759],
                [ 0.833250206225063,   0.004814348317439,  -0.999286510416273,  -0.633095739808868,  -0.368890743119366,  -0.633095739808868,  -0.999286510416273,   0.004814348317439],
                [ 0.063120922443392,   0.799934516166400,  -0.573061561512667,  -0.999286510416273,  -0.963805291282899,  -0.999286510416273,  -0.573061561512667,   0.799934516166400],
                [-0.995184514498978,   0.626977423649552,   0.799934516166400,   0.004814348317439,  -0.298096205735759,   0.004814348317439,   0.799934516166400,   0.626977423649552]
            ])
        ))


