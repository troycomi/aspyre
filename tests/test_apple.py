from unittest import TestCase
from tempfile import TemporaryDirectory
from copy import copy

from aspyre.apple.apple import Apple
from aspyre.apple.config import ApplePickerConfig

import os.path
DATA_DIR = os.path.join(os.path.dirname(__file__), 'saved_test_data', 'mrc_files')


class ApplePickerTestCase(TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def testPick(self):
        config = copy(ApplePickerConfig)
        config.particle_size = 78
        apple_picker = Apple(config, DATA_DIR)

        with TemporaryDirectory() as temp_dir:
            config.output_dir = temp_dir
            apple_picker.pick_particles()
