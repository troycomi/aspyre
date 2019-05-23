import numpy as np
import matplotlib.pyplot as plt

from cryo.source import SourceFilter
from cryo.source.simulation import Simulation
from cryo.imaging.filters import RadialCTFFilter

if __name__ == '__main__':
    sim = Simulation(
        L=64,
        n=1024,
        filters=SourceFilter(
            [RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)],
            n=1024
        )
    )

    images = sim.clean_images(0, 20)
    plt.imshow(images[:, :, 10], cmap='gray')
    plt.show()
