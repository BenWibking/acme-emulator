import numpy as np
import h5py

# cython import 'density' module
import pyximport; pyximport.install()
import density

def compute_densities(particles_filename, galaxies_filename, output_filename):
    particles_file = h5py.File(particles_filename)
    particles = particles_file['particles']

    galaxies_file = h5py.File(galaxies_filename)
    galaxies = galaxies_file['particles']

    # for computation
    zmin = 0.
    zmax = 400.
    mask = np.logical_and(particles['z'] < zmax, particles['z'] > zmin)

    particles_x = particles['x'][mask]
    particles_y = particles['y'][mask]

    gmask = np.logical_and(galaxies['z'] < zmax, galaxies['z'] > zmin)
    galaxies_x = galaxies['x'][gmask]
    galaxies_y = galaxies['y'][gmask]

    xlength = np.abs(np.max(particles_x) - np.min(particles_x))
    ylength = np.abs(np.max(particles_y) - np.min(particles_y))
    boxsize = np.max(np.array([xlength,ylength]))

    # for plotting only
    ngrid = 1024
    xmin = 0.
    xmax = 100.
    ymin = 0.
    ymax = 100.
    xl = int(xmin/boxsize * ngrid)
    xh = int(xmax/boxsize * ngrid)
    yl = int(ymin/boxsize * ngrid)
    yh = int(ymax/boxsize * ngrid)
    xmin = xl / ngrid * boxsize
    xmax = xh / ngrid * boxsize
    ymin = yl / ngrid * boxsize
    ymax = yh / ngrid * boxsize

    # set up Fourier transform, allocate arrays:
    hires_rhogrid = np.empty((ngrid,ngrid),dtype=np.float32)
    hires_rhogrid = np.array(density.cic_2d_grid(particles_x,particles_y,hires_rhogrid,ngrid,boxsize))
    hires_rhogrid = np.arctan(hires_rhogrid)

    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    axisfontsize=13
    ticklabelfontsize=13

    plt,ax = plt.subplots(1,1)

    # plot density field
    image = ax.imshow(hires_rhogrid, cmap=cm.get_cmap('gray'),
                      extent=[0.,boxsize,0.,boxsize],interpolation='bilinear')
    #                  vmin=-1.0, vmax=1.0)
    #divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.1)
    #plt.colorbar(image,cax=cax)

    # plot galaxies
    ax.scatter(galaxies_x, galaxies_y, color='red', s=0.5)

    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)
    ax.set_xlabel(r'x ($h^{-1}$ Mpc)',fontsize=axisfontsize)
    ax.set_ylabel(r'y ($h^{-1}$ Mpc)',fontsize=axisfontsize)
    ax.xaxis.set_tick_params(labelsize=ticklabelfontsize)
    ax.yaxis.set_tick_params(labelsize=ticklabelfontsize)

    plt.tight_layout()
    plt.savefig(output_filename)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_particles_filename')
    parser.add_argument('input_galaxies_filename')
    parser.add_argument('output_density_filename')
    args = parser.parse_args()

    compute_densities(args.input_particles_filename,
                      args.input_galaxies_filename,
                      args.output_density_filename)
