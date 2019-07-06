import configparser

# hack from: http://stackoverflow.com/questions/2819696/parsing-properties-file-in-python/2819788
class AbacusConfigFile(object):
	def __init__(self,filename):
	
		self.filename = filename
		self.cp = configparser.ConfigParser(strict=False)
		
		with open(self.filename, mode='r', encoding='utf-8') as f:
			file_string = '\n'.join( [i.strip() for i in f.read().splitlines()] )
			config_string = '[config]\n' + file_string
			
		self.cp.read_string(config_string)
		
		self.particleMass = self.cp.getfloat('config','ParticleMassHMsun')
		self.boxSize = self.cp.getfloat('config','boxsizehmpc')
		self.vel_to_kms = self.cp.getfloat('config','velzspace_to_kms')
		self.OmegaNow_m = self.cp.getfloat('config','OmegaNow_m')
		self.Omega_M = self.cp.getfloat('config','Omega_M')
		self.sigma_8 = self.cp.getfloat('config','sigma_8')
		self.H0 = self.cp.getfloat('config','H0')
		self.redshift = self.cp.getfloat('config','Redshift')
		self.ombh2 = self.cp.getfloat('config','ombh2')
		self.omch2 = self.cp.getfloat('config','omch2')
		self.w0 = self.cp.getfloat('config','w0')
		self.ns = self.cp.getfloat('config','ns')
