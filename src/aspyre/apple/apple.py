import logging
import glob
import os
import numpy as np
from concurrent import futures
from tqdm import tqdm

from aspyre.apple.picking import Picker
from aspyre import config
from aspyre.utils import ensure

logger = logging.getLogger(__name__)


class Apple:
    def __init__(self, mrc_dir, output_dir=None, create_jpg=False):

        self.particle_size = config.apple.particle_size
        self.query_image_size = config.apple.query_image_size
        self.max_particle_size = config.apple.max_particle_size or self.particle_size * 2
        self.min_particle_size = config.apple.min_particle_size or self.particle_size // 4
        self.minimum_overlap_amount = config.apple.minimum_overlap_amount or self.particle_size // 10
        self.container_size = config.apple.container_size
        self.n_threads = config.apple.n_threads
        self.output_dir = output_dir
        self.create_jpg = create_jpg
        self.mrc_dir = mrc_dir

        if self.query_image_size is None:
            query_image_size = np.round(self.particle_size * 2 / 3)
            query_image_size -= query_image_size % 4
            query_image_size = int(query_image_size)
    
            self.query_image_size = query_image_size

        q_box = (4000 ** 2) / (self.query_image_size ** 2) * 4
        self.tau1 = config.apple.tau1 or int(q_box * 0.03)
        self.tau2 = config.apple.tau2 or int(q_box * 0.3)

        if self.output_dir is not None and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.verify_input_values()

    def verify_input_values(self):
        ensure(1 <= self.max_particle_size <= 3000, "Max particle size must be in range [1, 3000]!")
        ensure(1 <= self.query_image_size <= 3000, "Query image size must be in range [1, 3000]!")
        ensure(5 <= self.particle_size < 3000, "Particle size must be in range [5, 3000]!")
        ensure(1 <= self.min_particle_size < 3000, "Min particle size must be in range [1, 3000]!")

        max_tau1_value = (4000 / self.query_image_size * 2) ** 2
        ensure(0 <= self.tau1 <= max_tau1_value, f"tau1 must be a in range [0, {max_tau1_value}]!")

        max_tau2_value = max_tau1_value
        ensure(0 <= self.tau2 <= max_tau2_value, f"tau2 must be in range [0, {max_tau2_value}]!")

        ensure(0 <= self.minimum_overlap_amount <= 3000, "overlap must be in range [0, 3000]!")

        # max container_size condition is (container_size_max * 2 + 200 > 4000), which is 1900
        ensure(self.particle_size <= self.container_size <= 1900,
               f"Container size must be within range [{self.particle_size}, 1900]!")

        ensure(self.particle_size >= self.query_image_size,
               f"Particle size ({self.particle_size}) must exceed query image size ({self.query_image_size})!")

    def pick_particles(self):

        filenames = [os.path.basename(file) for file in glob.glob('{}/*.mrc'.format(self.mrc_dir))]
        logger.info("converting {} mrc files..".format(len(filenames)))

        pbar = tqdm(total=len(filenames))
        with futures.ThreadPoolExecutor(self.n_threads) as executor:
            to_do = []
            for filename in filenames:
                future = executor.submit(self.process_micrograph, filename, False)
                to_do.append(future)

            for _ in futures.as_completed(to_do):
                pbar.update(1)
        pbar.close()

    def process_micrograph(self, filename, return_centers=True):
        ensure(filename.endswith('.mrc'), f"Input file doesn't seem to be an MRC format! ({filename})")

        # add path to filename
        filename = os.path.join(self.mrc_dir, filename)

        picker = Picker(self.particle_size, self.max_particle_size, self.min_particle_size, self.query_image_size,
                        self.tau1, self.tau2, self.minimum_overlap_amount, self.container_size, filename,
                        self.output_dir)

        # return .mrc file as a float64 array
        micro_img = picker.read_mrc()  # return a micrograph as an numpy array

        # compute score for query images
        score = picker.query_score(micro_img)  # compute score using normalized cross-correlations

        while True:
            # train SVM classifier and classify all windows in micrograph
            segmentation = picker.run_svm(micro_img, score)

            # If all windows are classified identically, update tau_1 or tau_2
            if np.all(segmentation):
                picker.tau2 += 500
            elif not np.any(segmentation):
                picker.tau1 += 500
            else:
                break

        # discard suspected artifacts
        segmentation = picker.morphology_ops(segmentation)

        # get particle centers, saving as necessary
        centers = picker.extract_particles(segmentation, self.create_jpg)

        if return_centers:
            return centers
