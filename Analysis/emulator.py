import numpy as np

def load_derivative_file(filename):
	table = np.loadtxt(filename,unpack=False)
	binmin, binmax, zeros, derivative = [table[:,i] for i in range(4)]
	return binmin,binmax,derivative

def emulate(fiducial_model,derivatives,params,fiducial_params):
        model = np.zeros(fiducial_model.shape)
        model[:] = fiducial_model
        for i in range(params.shape[0]):
                model += (params[i] - fiducial_params[i]) * derivatives[i]
        return model

def emulator_data_from_files(fiducial_file, deriv_filenames):
        """compute the emulated observable from the linear model
        defined by the fiducial model in 'fiducial_file'
        and the derivatives defined in 'deriv_files'

        deriv_files should be a list of tuples, where the first part of the tuple
        is the filename, and the second part is the parameter value to emulate, and
        the third part is the parameter value of the fiducial model"""

        binmin,binmax,fiducial_model = load_derivative_file(fiducial_file)
        derivatives = []

        # read files into arrays
        for i, filename in enumerate(deriv_filenames):
                deriv_binmin, deriv_binmax, derivative = load_derivative_file(filename)
                if np.allclose(deriv_binmin,binmin) and np.allclose(deriv_binmax,binmax):
                        pass
                else:
                        raise Exception("bins do not match the fiducial model!")

                derivatives.append(derivative)

        return binmin,binmax,fiducial_model,derivatives
