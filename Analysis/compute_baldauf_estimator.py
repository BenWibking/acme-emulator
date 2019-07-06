import numpy as np

def compute_upsilon(binmin,binmax,wp,r0=None):
    """compute Baldauf estimator for a projected correlation function.
    Removes information from scales below r0."""
    
    # find bin centers
    r = 0.5*(binmin+binmax)

    # linearly interpolate binned wp to find wp(r0)
    ln_r = np.log(r)
    ln_wp = np.log(wp)
    ln_r0 = np.log(r0)
    ln_wp_r0 = np.interp(ln_r0, ln_r, ln_wp)
    wp_r0 = np.exp(ln_wp_r0)

    # compute upsilon(r) = wp(r) - (r0/r)**2 * wp(r0)
    upsilon = np.zeros(wp.shape)
    for i in range(upsilon.shape[0]):
        upsilon[i] = wp[i] - (r0/r[i])**2 * wp_r0
    
    upsilon[r < r0] = 0.
    return upsilon

def upsilon_from_files(filename,output_file,r0=None):
    binmin,binmax,null,wp = np.loadtxt(filename,unpack=True)
    upsilon = compute_upsilon(binmin,binmax,wp,r0=r0) # scales below r0 set to zero
    
    np.savetxt(output_file, np.c_[binmin, binmax, np.zeros(upsilon.shape[0]), upsilon],
               delimiter='\t')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument('output_file')
    parser.add_argument('--r0',type=float,default=5.0) # [h^-1 Mpc] eliminate information below this scale
    args = parser.parse_args()

    upsilon_from_files(args.input_file, args.output_file,
                       r0=args.r0)
