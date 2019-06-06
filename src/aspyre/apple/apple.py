import logging
import glob
import os
import numpy as np
from concurrent import futures
from tqdm import tqdm

from aspyre.apple.exceptions import ConfigError
from aspyre.apple.picking import Picker
from aspyre import config

logger = logging.getLogger(__name__)


class Apple:
    """ This class is the layer between the user and the picking algorithm. """

    def __init__(self, mrc_dir):

        self.particle_size = config.apple.particle_size
        self.query_image_size = config.apple.query_image_size
        self.max_particle_size = config.apple.max_particle_size
        self.min_particle_size = config.apple.min_particle_size
        self.minimum_overlap_amount = config.apple.minimum_overlap_amount
        self.tau1 = config.apple.tau1
        self.tau2 = config.apple.tau2
        self.container_size = config.apple.container_size
        self.proc = config.apple.proc
        self.output_dir = config.apple.output_dir
        self.create_jpg = config.apple.create_jpg
        self.mrc_dir = mrc_dir

        # set default values if needed
        if self.query_image_size is None:
            query_window_size = np.round(self.particle_size * 2 / 3)
            query_window_size -= query_window_size % 4
            query_window_size = int(query_window_size)
    
            self.query_image_size = query_window_size

        if self.max_particle_size is None:
            self.max_particle_size = self.particle_size * 2

        if self.min_particle_size is None:
            self.min_particle_size = int(self.particle_size / 4)

        if self.minimum_overlap_amount is None:
            self.minimum_overlap_amount = int(self.particle_size / 10)

        if self.output_dir is None:
            abs_path = os.path.abspath(self.mrc_dir)
            self.output_dir = os.path.join(os.path.dirname(abs_path), 'star_dir')
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

        q_box = (4000 ** 2) / (self.query_image_size ** 2) * 4
        if self.tau1 is None:
            self.tau1 = int(q_box * 3 / 100)

        if self.tau2 is None:
            self.tau2 = int(q_box * 30 / 100)

        self.verify_input_values()
        self.print_values()

    def print_values(self):
        """Printing all parameters to screen."""

        try:
            std_out_width = os.get_terminal_size().columns
        except OSError:
            std_out_width = 100

        logger.info(' Parameter Report '.center(std_out_width, '=') + '\n')

        params = ['particle_size',
                  'query_image_size',
                  'max_particle_size',
                  'min_particle_size',
                  'minimum_overlap_amount',
                  'tau1',
                  'tau2',
                  'container_size',
                  'proc',
                  'output_dir']

        for param in params:
            logger.info('%(param)-40s %(value)-10s' % {"param": param, "value": getattr(self, param)})

        logger.info('\n' + ' Progress Report '.center(std_out_width, '=') + '\n')

    def verify_input_values(self):
        """Verify parameter values make sense.
        
        Sanity check for the attributes of this instance of the Apple class.
        
        Raises:
            ConfigError: Attribute is out of range.
        """
        
        if not 1 <= self.max_particle_size <= 3000:
            raise ConfigError("Error", "Max particle size must be in range [1, 3000]!")

        if not 1 <= self.query_image_size <= 3000:
            raise ConfigError("Error", "Query image size must be in range [1, 3000]!")

        if not 5 <= self.particle_size < 3000:
            raise ConfigError("Error", "Particle size must be in range [5, 3000]!")

        if not 1 <= self.min_particle_size < 3000:
            raise ConfigError("Error", "Min particle size must be in range [1, 3000]!")

        max_tau1_value = (4000 / self.query_image_size * 2) ** 2
        if not 0 <= self.tau1 <= max_tau1_value:
            raise ConfigError("Error",
                              "\u03C4\u2081 must be a in range [0, {}]!".format(max_tau1_value))

        max_tau2_value = (4000 / self.query_image_size * 2) ** 2
        if not 0 <= self.tau2 <= max_tau2_value:
            raise ConfigError("Error",
                              "\u03C4\u2082 must be in range [0, {}]!".format(max_tau2_value))

        if not 0 <= self.minimum_overlap_amount <= 3000:
            raise ConfigError("Error", "overlap must be in range [0, 3000]!")

        # max container_size condition is (conainter_size_max * 2 + 200 > 4000), which is 1900
        if not self.particle_size <= self.container_size <= 1900:
            raise ConfigError("Error", "Container size must be within range [{}, 1900]!".format(
                self.particle_size))

        if self.particle_size < self.query_image_size:
            raise ConfigError("Error",
                              "Particle size must exceed query image size! particle size:{}, "
                              "query image size: {}".format(self.particle_size,
                                                            self.query_image_size))

        if self.proc < 1:
            raise ConfigError("Error", "Please select at least one processor!")

    def pick_particles(self):

        filenames = [os.path.basename(file) for file in glob.glob('{}/*.mrc'.format(self.mrc_dir))]
        logger.info("converting {} mrc files..".format(len(filenames)))

        pbar = tqdm(total=len(filenames))
        with futures.ProcessPoolExecutor(self.proc) as executor:
            to_do = []
            for filename in filenames:
                future = executor.submit(self.process_micrograph, filename)
                to_do.append(future)

            for _ in futures.as_completed(to_do):
                pbar.update(1)
        pbar.close()

    def process_micrograph(self, filename):
        """Pick particles.
        
        Implements the APPLE picker algorithm (Heimowitz, Andén and Singer,
        "APPLE picker: Automatic particle picking, a low-effort cryo-EM framework").
        
        Args:
            filename: Name of micrograph for picking.
            
            Raises:
                ConfigError: Incorrect format for micrograph file.
        """

        if not filename.endswith('.mrc'):
            raise ConfigError("Input file doesn't seem to be an MRC format! ({})".format(filename))

        # add path to filename
        filename = os.path.join(self.mrc_dir, filename)

        picker = Picker(self.particle_size, self.max_particle_size, self.min_particle_size, self.query_image_size,
                        self.tau1, self.tau2, self.minimum_overlap_amount, self.container_size, filename,
                        self.output_dir)

        # update user
        logger.info('Processing {}..'.format(os.path.basename(filename)))

        # return .mrc file as a float64 array
        micro_img = picker.read_mrc()  # return a micrograph as an numpy array

        # compute score for query images
        score = picker.query_score(micro_img)  # compute score using normalized cross-correlations

        tau1 = self.tau1
        tau2 = self.tau2

        while True:
            # train SVM classifier and classify all windows in micrograph
            segmentation = picker.run_svm(micro_img, score)

            # If all windows are classified identically, update tau_1 or tau_2
            if np.array_equal(segmentation,
                              np.ones((segmentation.shape[0], segmentation.shape[1]))):
                tau2 = tau2 + 500

            elif np.array_equal(segmentation,
                                np.zeros((segmentation.shape[0], segmentation.shape[1]))):
                tau1 = tau1 + 500

            else:
                break

        # discard suspected artifacts
        segmentation = picker.morphology_ops(segmentation)

        # create output star file
        centers = picker.extract_particles(segmentation)
        
        if config.apple.create_jpg:
            picker.display_picks(centers)

        return centers
