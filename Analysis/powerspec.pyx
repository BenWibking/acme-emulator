#!python
#cython: boundscheck=False
#cython: wraparound=False

import numpy as np
import pyfftw as fft

#from cython.parallel import parallel,prange
from libc.math cimport sin, cos, sqrt
from libc.stdint cimport int64_t

cdef extern from "math.h":
	double M_PI

cdef extern from "complex.h" nogil:
	double complex cexp(double complex)

# fourier grid computation
cdef int num_threads = 8


def plan_fft(ngrid):
	# do an in-place transform! (this does not change the array indexing, only padding)
	input_size = (ngrid,ngrid,ngrid)
	padded_size = (input_size[0],input_size[1],2*(input_size[2]//2 + 1))
	full_array = fft.empty_aligned(padded_size,dtype='float32',order='C')

	rhogrid = full_array[:, :, :input_size[2]]
	fft_of_rhogrid = full_array.view('complex64')

	fft_plan = fft.FFTW(rhogrid,fft_of_rhogrid,axes=(0,1,2),
						threads=num_threads,direction='FFTW_FORWARD',
						flags=('FFTW_ESTIMATE',))

	return rhogrid, fft_of_rhogrid, fft_plan


def plan_fft_double(ngrid):
	# do an in-place transform! (this does not change the array indexing, only padding)
	input_size = (ngrid, ngrid, ngrid)
	padded_size = (input_size[0], input_size[1], 2*(input_size[2]//2 + 1))
	full_array = fft.empty_aligned(padded_size,dtype='float64',order='C')

	rhogrid = full_array[:, :, :input_size[2]]
	fft_of_rhogrid = full_array.view('complex128')

	fft_plan = fft.FFTW(rhogrid,fft_of_rhogrid,axes=(0,1,2),
						threads=num_threads,direction='FFTW_FORWARD',
						flags=('FFTW_ESTIMATE',))

	return rhogrid, fft_of_rhogrid, fft_plan


cpdef void boxwrap(float[:] pos, float boxsize) nogil:
	cdef Py_ssize_t i,j
	for i in xrange(pos.shape[0]):
		if (pos[i] > boxsize):
			pos[i] -= boxsize
		elif (pos[i] < 0.):
			pos[i] += boxsize


cpdef void boxwrap_double(double[:] pos, double boxsize) nogil:
	cdef Py_ssize_t i,j
	for i in xrange(pos.shape[0]):
		if (pos[i] > boxsize):
			pos[i] -= boxsize
		elif (pos[i] < 0.):
			pos[i] += boxsize


cpdef void cic_grid(float[:] xpos, float[:] ypos, float[:] zpos,
					float[:,:,:] rhogrid, int ngrid, float boxsize) nogil:

	""" return a binning of densities according to cloud-in-cell binning. """

	cdef double mass = (<double>ngrid)**3 / (<double>xpos.shape[0]) # this is actually a density!!
	cdef double slab_fac = (<double>ngrid) / boxsize
	cdef double half_dx = boxsize / (2.0 * <double>ngrid)
	cdef int ix,iy,iz,iix,iiy,iiz
	cdef double dx,dy,dz,x,y,z
	cdef Py_ssize_t i,j

	rhogrid[:,:,:] = -1.0

	for i in xrange(xpos.shape[0]):
		x = xpos[i]
		y = ypos[i]
		z = zpos[i]
		
		ix = <int>(slab_fac * x)
		iy = <int>(slab_fac * y)
		iz = <int>(slab_fac * z)

		iix = (ix + 1) % ngrid
		iiy = (iy + 1) % ngrid
		iiz = (iz + 1) % ngrid

		dx = slab_fac * x - <double>ix
		dy = slab_fac * y - <double>iy
		dz = slab_fac * z - <double>iz

		ix = ix % ngrid
		iy = iy % ngrid
		iz = iz % ngrid

		rhogrid[ix,iy,iz]     += (1.-dx) * (1.-dy) * (1.-dz) * mass
		rhogrid[ix,iy,iiz]   += (1.-dx) * (1.-dy) * dz * mass
		rhogrid[ix,iiy,iz]   += (1.-dx) * dy * (1.-dz) * mass
		rhogrid[ix,iiy,iiz] += (1.-dx) * dy * dz * mass

		rhogrid[iix,iy,iz]     += dx * (1.-dy) * (1.-dz) * mass
		rhogrid[iix,iy,iiz]   += dx * (1.-dy) * dz * mass
		rhogrid[iix,iiy,iz]   += dx * dy * (1.-dz) * mass
		rhogrid[iix,iiy,iiz] += dx * dy * dz * mass


cpdef void cic_grid_weights(double[:] xpos, double[:] ypos, double[:] zpos, double[:] weights, 
							double[:,:,:] rhogrid, int ngrid, double boxsize) nogil:

	""" return a binning of densities according to cloud-in-cell binning.
		(N.B. when using weights it is best to use double precision.) """

	cdef double slab_fac = (<double>ngrid) / boxsize
	cdef double half_dx = boxsize / (2.0 * <double>ngrid)
	cdef int ix,iy,iz,iix,iiy,iiz
	cdef double dx,dy,dz,x,y,z,weight,sum_weights
	cdef Py_ssize_t i,j

	rhogrid[:,:,:] = -1.0

	sum_weights = 0.
	for i in xrange(xpos.shape[0]):
		sum_weights += weights[i]

	cdef double mass = (<double>ngrid)**3 / sum_weights # this is actually a density!!

	for i in xrange(xpos.shape[0]):
		x = xpos[i]
		y = ypos[i]
		z = zpos[i]
		weight = weights[i]
		
		ix = <int>(slab_fac * x)
		iy = <int>(slab_fac * y)
		iz = <int>(slab_fac * z)

		iix = (ix + 1) % ngrid
		iiy = (iy + 1) % ngrid
		iiz = (iz + 1) % ngrid

		dx = slab_fac * x - <double>ix
		dy = slab_fac * y - <double>iy
		dz = slab_fac * z - <double>iz

		ix = ix % ngrid
		iy = iy % ngrid
		iz = iz % ngrid

		rhogrid[ix,iy,iz]     += (1.-dx) * (1.-dy) * (1.-dz) * mass * weight
		rhogrid[ix,iy,iiz]   += (1.-dx) * (1.-dy) * dz * mass * weight
		rhogrid[ix,iiy,iz]   += (1.-dx) * dy * (1.-dz) * mass * weight
		rhogrid[ix,iiy,iiz] += (1.-dx) * dy * dz * mass * weight

		rhogrid[iix,iy,iz]     += dx * (1.-dy) * (1.-dz) * mass * weight
		rhogrid[iix,iy,iiz]   += dx * (1.-dy) * dz * mass * weight
		rhogrid[iix,iiy,iz]   += dx * dy * (1.-dz) * mass * weight
		rhogrid[iix,iiy,iiz] += dx * dy * dz * mass * weight


cpdef void brute_force_ft_particles(float[:] xpos, float[:] ypos, float[:] zpos, float complex [:,:,:] ftgrid, float boxsize):
	"""compute the Fourier transform of the input particle distribution by brute force."""
	cdef Py_ssize_t p,i,j,k,ngrid,ieff,jeff,keff,Np,knyq,k_idx
	cdef double kx,ky,kz,dot_product,k2
	cdef double complex kernel,norm

	Np = xpos.shape[0]
	ngrid = ftgrid.shape[0]
	knyq = ngrid//2
	cdef float mass = 1.0 / <double>Np # this is actually a mass! no factors of ngrid!!
	ftgrid[:,:,:] = 0.0

	for i in xrange(ngrid):
		print(i)
		with nogil:
			for j in xrange(ngrid):
				for k in xrange(ngrid//2 + 1):
					ieff = i
					jeff = j
					keff = k
					if(i > ngrid//2):
						ieff -= ngrid
					if(j > ngrid//2):
						jeff -= ngrid
					if(k > ngrid//2):
						keff -= ngrid

					kx = (<double>ieff)
					ky = (<double>jeff)
					kz = (<double>keff)
					k2 = kx*kx + ky*ky + kz*kz

					k_idx = <int> (sqrt(k2) + 0.5)
					if(k_idx <= knyq):
						kx = <double>ieff*2.0*M_PI/boxsize
						ky = <double>jeff*2.0*M_PI/boxsize
						kz = <double>keff*2.0*M_PI/boxsize

						for p in xrange(Np):
							dot_product = kx*xpos[p] + ky*ypos[p] + kz*zpos[p]
							ftgrid[i,j,k] += mass * cexp(1j*dot_product)


def fft_grid(float complex [:,:,:] fft_of_rhogrid, fft_forward_plan):
	# compute the fft
	fft_forward_plan.execute()

	cdef Py_ssize_t i,j,k,ngrid,ieff,jeff,keff
	cdef double kx,ky,kz,kx2,ky2,kz2,k2,sinc,w1,w2,w3
	cdef double complex kernel,norm

	ngrid = fft_of_rhogrid.shape[0]

	# deconvolve window function
	with nogil:
		for i in xrange(ngrid):
			for j in xrange(ngrid):
				for k in xrange(ngrid//2 + 1):
					ieff = i
					jeff = j
					keff = k
					if(i > ngrid//2):
						ieff -= ngrid
					if(j > ngrid//2):
						jeff -= ngrid
					if(k > ngrid//2):
						keff -= ngrid
				
					kx = (<double>ieff)
					ky = (<double>jeff)
					kz = (<double>keff)
					kx2 = kx*kx
					ky2 = ky*ky
					kz2 = kz*kz
					k2 = kx2 + ky2 + kz2
					
					if(i != 0):
						kx *= M_PI/<double>ngrid
						w1 = sin(kx)/kx
					else:
						w1 = 1. # lim x->0 sin(x)/x = 1

					if(j != 0):
						ky *= M_PI/<double>ngrid
						w2 = sin(ky)/ky
					else:
						w2 = 1.

					if(k != 0):
						kz *= M_PI/<double>ngrid
						w3 = sin(kz)/kz
					else:
						w3 = 1.

					sinc = w1 * w2 * w3
					# sinc^2 deconvolves CIC assignment
					norm = (<double>ngrid)**3 # undo fftw normalization
					kernel = (sinc*sinc)*norm
					fft_of_rhogrid[i,j,k] = fft_of_rhogrid[i,j,k] / kernel


def fft_grid_double(double complex [:,:,:] fft_of_rhogrid, fft_forward_plan):
	# compute the fft
	fft_forward_plan.execute()

	cdef Py_ssize_t i,j,k,ngrid,ieff,jeff,keff
	cdef double kx,ky,kz,kx2,ky2,kz2,k2,sinc,w1,w2,w3
	cdef double complex kernel,norm

	ngrid = fft_of_rhogrid.shape[0]

	# deconvolve window function
	with nogil:
		for i in xrange(ngrid):
			for j in xrange(ngrid):
				for k in xrange(ngrid//2 + 1):
					ieff = i
					jeff = j
					keff = k
					if(i > ngrid//2):
						ieff -= ngrid
					if(j > ngrid//2):
						jeff -= ngrid
					if(k > ngrid//2):
						keff -= ngrid
				
					kx = (<double>ieff)
					ky = (<double>jeff)
					kz = (<double>keff)
					kx2 = kx*kx
					ky2 = ky*ky
					kz2 = kz*kz
					k2 = kx2 + ky2 + kz2
					
					if(i != 0):
						kx *= M_PI/<double>ngrid
						w1 = sin(kx)/kx
					else:
						w1 = 1. # lim x->0 sin(x)/x = 1

					if(j != 0):
						ky *= M_PI/<double>ngrid
						w2 = sin(ky)/ky
					else:
						w2 = 1.

					if(k != 0):
						kz *= M_PI/<double>ngrid
						w3 = sin(kz)/kz
					else:
						w3 = 1.

					sinc = w1 * w2 * w3
					# sinc^2 deconvolves CIC assignment
					norm = (<double>ngrid)**3 # undo fftw normalization
					kernel = (sinc*sinc)*norm
					fft_of_rhogrid[i,j,k] = fft_of_rhogrid[i,j,k] / kernel


def power_grid(float complex [:,:,:] fft_of_rhogrid):
	cdef Py_ssize_t i,j,k,ngrid,ieff,jeff,keff,kmax,k_idx
	cdef double kx,ky,kz,kx2,ky2,kz2,k2,sinc,w1,w2,w3,fac

	ngrid = fft_of_rhogrid.shape[0]
	knyq = ngrid//2
	fac = 1.0
	kmax = knyq
	npk = np.zeros(kmax+1,dtype=np.float64)
	nnmodes = np.zeros(kmax+1,dtype=np.int64)
	ksph = np.linspace(0., float(kmax), kmax+1, dtype=np.float64)
	cdef double [:] pk = npk
	cdef int64_t [:] nmodes = nnmodes

	# compute angle-averaging
	with nogil:
		for i in xrange(ngrid):
			for j in xrange(ngrid):
				for k in xrange(ngrid//2 + 1):
					ieff = i
					jeff = j
					keff = k
					if(i > ngrid//2):
						ieff -= ngrid
					if(j > ngrid//2):
						jeff -= ngrid
					if(k > ngrid//2):
						keff -= ngrid
				
					kx = (<double>ieff)
					ky = (<double>jeff)
					kz = (<double>keff)
					kx2 = kx*kx
					ky2 = ky*ky
					kz2 = kz*kz                    
					k2 = kx2 + ky2 + kz2

					# avoid double-counting due to discrete symmetries (check this with 2LPTic...)
					#if(k==0 and not (i == ngrid//2 or j == ngrid//2 or (i==0 and j==0))):
					#    fac = 0.5

					k_idx = <int> (sqrt(k2) + 0.5)
					if(k_idx <= knyq):
						pk[k_idx] += fft_of_rhogrid[i,j,k].real**2 + fft_of_rhogrid[i,j,k].imag**2
						nmodes[k_idx] += 1

	# divide by nmodes
	for i in xrange(1,pk.shape[0]):
		pk[i] /= <double>(nmodes[i])

	return ksph[1:],pk[1:],nmodes[1:]


def power_grid_double(double complex [:,:,:] fft_of_rhogrid):
	cdef Py_ssize_t i,j,k,ngrid,ieff,jeff,keff,kmax,k_idx
	cdef double kx,ky,kz,kx2,ky2,kz2,k2,sinc,w1,w2,w3,fac

	ngrid = fft_of_rhogrid.shape[0]
	knyq = ngrid//2
	fac = 1.0
	kmax = knyq
	npk = np.zeros(kmax+1,dtype=np.float64)
	nnmodes = np.zeros(kmax+1,dtype=np.int64)
	ksph = np.linspace(0., float(kmax), kmax+1, dtype=np.float64)
	cdef double [:] pk = npk
	cdef int64_t [:] nmodes = nnmodes

	# compute angle-averaging
	with nogil:
		for i in xrange(ngrid):
			for j in xrange(ngrid):
				for k in xrange(ngrid//2 + 1):
					ieff = i
					jeff = j
					keff = k
					if(i > ngrid//2):
						ieff -= ngrid
					if(j > ngrid//2):
						jeff -= ngrid
					if(k > ngrid//2):
						keff -= ngrid
				
					kx = (<double>ieff)
					ky = (<double>jeff)
					kz = (<double>keff)
					kx2 = kx*kx
					ky2 = ky*ky
					kz2 = kz*kz                    
					k2 = kx2 + ky2 + kz2

					# avoid double-counting due to discrete symmetries (check this with 2LPTic...)
					#if(k==0 and not (i == ngrid//2 or j == ngrid//2 or (i==0 and j==0))):
					#    fac = 0.5

					k_idx = <int> (sqrt(k2) + 0.5)
					if(k_idx <= knyq):
						pk[k_idx] += fft_of_rhogrid[i,j,k].real**2 + fft_of_rhogrid[i,j,k].imag**2
						nmodes[k_idx] += 1

	# divide by nmodes
	for i in xrange(1,pk.shape[0]):
		pk[i] /= <double>(nmodes[i])

	return ksph[1:],pk[1:],nmodes[1:]



