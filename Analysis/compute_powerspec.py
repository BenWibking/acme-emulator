import numpy as np
import h5py

# cython import 'powerspec' module
import pyximport; pyximport.install()
import powerspec

def power_from_particles(particles_x,particles_y,particles_z,ngrid,boxsize,
                         rhogrid,fft_of_rhogrid,fft_plan,subtract_shot_noise=True):
    # compute density field
    print('\tcomputing density field...',end='',flush=True)
    powerspec.cic_grid(particles_x,particles_y,particles_z,rhogrid,ngrid,boxsize)
    print('done.')
    # compute FFT
    print('\tcomputing FFT...',end='',flush=True)
    powerspec.fft_grid(fft_of_rhogrid, fft_plan)
    print('done.')
    # compute angle-averaging
    print('\tcomputing angle-averaging...',end='',flush=True)
    k, pk, nmodes = powerspec.power_grid(fft_of_rhogrid)
    print('done.')
    k = np.array(k)
    pk = np.array(pk)
    nmodes = np.array(nmodes)
    # convert to physically-meaningful k
    k *= (2.0*np.pi/boxsize)
    if subtract_shot_noise == True:
        # subtract shot noise (*can be dangerous!*)
        pk -= 1.0/float(particles_x.shape[0])
    # normalize volume
    pk *= (boxsize**3)
    # remove strongly-aliased region of k-space
    frac_nyquist_max = 0.5
    kmax = frac_nyquist_max*ngrid*(np.pi/boxsize)
    return k[k<=kmax],pk[k<=kmax],nmodes[k<=kmax]


def power_from_particles_weights(particles_x,particles_y,particles_z,weights,ngrid,boxsize,
                         rhogrid,fft_of_rhogrid,fft_plan,subtract_shot_noise=True):

    # compute density field
    print('\tcomputing density field...',end='',flush=True)
    powerspec.cic_grid_weights(particles_x,particles_y,particles_z,weights,rhogrid,ngrid,boxsize)
    print('done.')

    # compute FFT
    print('\tcomputing FFT...',end='',flush=True)
    powerspec.fft_grid_double(fft_of_rhogrid, fft_plan)
    print('done.')

    # compute angle-averaging
    print('\tcomputing angle-averaging...',end='',flush=True)
    k, pk, nmodes = powerspec.power_grid_double(fft_of_rhogrid)
    print('done.')
    k = np.array(k)
    pk = np.array(pk)
    nmodes = np.array(nmodes)

    # convert to physically-meaningful k
    k *= (2.0*np.pi/boxsize)

    if subtract_shot_noise == True:
        # subtract shot noise (*can be dangerous!*)
        pk -= 1.0/float(particles_x.shape[0])

    # normalize volume
    pk *= (boxsize**3)

    # remove strongly-aliased region of k-space
    frac_nyquist_max = 0.5
    kmax = frac_nyquist_max*ngrid*(np.pi/boxsize)

    return k[k<=kmax],pk[k<=kmax],nmodes[k<=kmax]


def fold_particles(particles_x,particles_y,particles_z,boxsize,fold=0):
    rescale = 2.0**(fold)
    powerspec.boxwrap(particles_x,boxsize/rescale)
    powerspec.boxwrap(particles_y,boxsize/rescale)
    powerspec.boxwrap(particles_z,boxsize/rescale)
    particles_x *= rescale
    particles_y *= rescale
    particles_z *= rescale


def read_particles(particles_filename):
    particles_file = h5py.File(particles_filename)
    particles = particles_file['particles']
    particles_x = particles['x']
    particles_y = particles['y']
    particles_z = particles['z']

    xlength = np.abs(np.max(particles_x) - np.min(particles_x))
    ylength = np.abs(np.max(particles_y) - np.min(particles_y))
    zlength = np.abs(np.max(particles_z) - np.min(particles_z))
    boxsize = np.max(np.array([xlength,ylength,zlength]))

    return particles_x,particles_y,particles_z,boxsize


def compute_power(particles_filename, output_filename, ngrid=512, nfolds=5):
    """compute power spectrum from particles."""

    kmax_prev = 0.
    k=np.array([])
    pk=np.array([])
    nmodes=np.array([])

    # allocate arrays, plan FFT
    print('setting up FFT...',end='',flush=True)
    rhogrid, fft_of_rhogrid, fft_plan = powerspec.plan_fft(ngrid)
    print('done.')

    for i in range(nfolds):
        print("computing fold %d" % i)

        # (re-)read particles, fold particles
        print('\treading particles...',end='',flush=True)
        particles_x,particles_y,particles_z,boxsize = read_particles(particles_filename)
        fold_particles(particles_x,particles_y,particles_z,boxsize,fold=i)
        print('done.')

        # compute power spectrum for this folding
        this_k, this_pk, this_nmodes = power_from_particles(particles_x,particles_y,particles_z,ngrid,boxsize,rhogrid,fft_of_rhogrid,fft_plan)
        this_k *= 2.0**i

        # concatenate this_k, this_pk, this_nmodes to k, pk, nmodes if k>kmax_prev
        mask = this_k > kmax_prev
        k = np.concatenate((k,this_k[mask]))
        pk = np.concatenate((pk,this_pk[mask]))
        nmodes = np.concatenate((nmodes,this_nmodes[mask]))
        kmax_prev = this_k[-1]

    # save to file
    #for i in range(pk.shape[0]):
    #    print('k=%s, pk=%g, nmodes=%s' % (k[i], pk[i], nmodes[i]))
    print('saving to file...',end='',flush=True)
    np.savetxt(output_filename, np.c_[k, pk])
    print('done.')

    # compute brute-force
#    powerspec.brute_force_ft_particles(particles_x,particles_y,particles_z,fft_of_rhogrid,boxsize)
#    k_bf, pk_bruteforce, nmodes_bf = powerspec.power_grid(fft_of_rhogrid, boxsize)
#    pk_bruteforce *= (boxsize**3)
#    np.savetxt(output_filename + '.brute_force', np.c_[k_bf, pk_bruteforce])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_particles_filename')
    parser.add_argument('output_pk_filename')
    args = parser.parse_args()

    compute_power(args.input_particles_filename,args.output_pk_filename)
