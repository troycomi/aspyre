import numpy as np
import matplotlib.pyplot as plt

from cryo.source import SourceFilter
from cryo.source.simulation import Simulation
from cryo.imaging.filters import RadialCTFFilter

if __name__ == '__main__':

    sim = Simulation(
        L=32,
        n=1024,
        filters=SourceFilter(
            [RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 7)],
            n=1024
        )
    )

    vol = sim.vols[:, :, :, 0]
    print(vol.shape)

    # Visualize volume
    L = vol.shape[0]
    x, y, z = np.meshgrid(np.arange(L), np.arange(L), np.arange(L))
    ax = plt.axes(projection='3d')
    vol = (vol - np.min(vol))/(np.max(vol)-np.min(vol))
    cmap = plt.get_cmap("hot_r")
    ax.scatter3D(x, y, z, c=vol.flatten(), cmap=cmap)
    plt.show()
