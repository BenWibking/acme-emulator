from boxsdk import JWTAuth, Client
from boxsdk.object.folder import Folder
from pathlib import PurePosixPath, Path
import os
import os.path
import pycurl
import progressbar


## Configuration ##

# Set the path to your JWT app config JSON file here!
pathToConfigJson = "boxapi_config.json"

# Set the path to a folder you'd like to traverse here!
shared_link_url = "https://osu.box.com/v/acme-simulations-reduced"

# Set the path to the local download folder
local_download_path = "./AbacusCosmos"


## Functions ##

def get_authenticated_client(configPath):
	"""Get an authenticated Box client for a JWT service account
	
	Arguments:
		configPath {str} -- Path to the JSON config file for your Box JWT app
	
	Returns:
		Client -- A Box client for the JWT service account

	Raises:
		ValueError -- if the configPath is empty or cannot be found.
	"""
	if (os.path.isfile(configPath) == False):
		raise ValueError(f"configPath must be a path to the JSON config file for your Box JWT app")
	auth = JWTAuth.from_settings_file(configPath)
	print("Authenticating...")
	auth.authenticate_instance()
	return Client(auth)


def get_path(item, truncate_prefix=0):
	parents = map(lambda p: p['name'], item.path_collection['entries'][truncate_prefix:])
	path = f"{'/'.join(parents)}/{item.name}".strip('/')
	return path


def print_path(item):
	"""Print the ID and path of a given Box file or folder."""
	item_id = item.id.rjust(12, ' ')
	path = get_path(item)
	print(f"{item_id} /{path}")

	
def get_subitems(folder, fields = ["id","name","path_collection","size"]):
	"""Get a collection of all immediate folder items
	
	Arguments:
		folder {Folder} -- The Box folder whose contents we want to fetch
	
	Keyword Arguments:
		fields {list} -- An optional list of fields to include with each item (default: {["id","name","path_collection"]})
	
	Returns:
		list -- A collection of Box files and folders.
	"""
	return folder.get_items(fields=fields)


def print_user_info(client):
	"""Print the name and login of the current authenticated Box user
		
	Arguments:
		client {Client} -- An authenticated Box client
	"""
	user = client.user('me').get()
	print("")
	print("Authenticated User")
	print(f"Name: {user.name}")
	print(f"Login: {user.login}")

	
def download_file(file, download_root):
	"""Download a Box file, saving it with its 'intrinsic' path inside local_root.

	Arguments:
		file {File} -- the Box file to download.
	"""

	## determine remote url
	data_product_url = file.get_download_url()

	## determine local download location
	download_path = download_root / PurePosixPath(get_path(file, truncate_prefix=1))

	print(f"download to: {download_path}")

	if not os.path.exists(download_path.parent):
		os.makedirs(download_path.parent)
	
	## check if file is already completely downloaded:
	c = pycurl.Curl()
	c.setopt(c.URL, data_product_url)
	c.setopt(c.NOBODY, 1) # get headers only
	c.perform()
	remote_filesize = c.getinfo(c.CONTENT_LENGTH_DOWNLOAD)
						
	if (os.path.exists(download_path) and os.path.getsize(download_path) == remote_filesize):
		print("\tSkipping file, already downloaded.")

	else:
		curl = pycurl.Curl()
		curl.setopt(pycurl.URL, data_product_url)
		curl.setopt(pycurl.FOLLOWLOCATION, 1)
		curl.setopt(pycurl.MAXREDIRS, 5)
		curl.setopt(pycurl.CONNECTTIMEOUT, 30)
							
		if os.path.exists(download_path):
				fp = open(download_path, 'ab')
				curl.setopt(pycurl.RESUME_FROM, os.path.getsize(download_path))
		else:
				fp = open(download_path, 'wb')
							
		bar = progressbar.DataTransferBar(max_value=remote_filesize)
			
		initial_size = os.path.getsize(download_path)
							
		def progress(total, existing, upload_t, upload_d):
				downloaded = existing + initial_size
				bar.update(downloaded)
															
		curl.setopt(pycurl.NOPROGRESS, 0)
		curl.setopt(pycurl.PROGRESSFUNCTION, progress)
		curl.setopt(pycurl.WRITEDATA, fp)
							
		print("\tdownloading...")
		curl.perform()
	
		curl.close()
		fp.close()
		bar.finish()

	print(f"")


def walk_folder_tree(folder, download_root):
	"""Traverse a Box folder tree, performing the specified action on every file and folder.

	Arguments:
		folder {Folder} -- The Box folder to traverse.
	"""

	subitems = list(get_subitems(folder))
	
	for file in filter(lambda i: i.type=="file", subitems):
		print_path(file)
		download_file(file, download_root)

	for subfolder in filter(lambda i: i.type=="folder", subitems):
		walk_folder_tree(subfolder, download_root)

		
def print_folder_tree(folder, download_root):
	"""Print the contents of a Box folder tree
	
	Arguments:
		folder {Folder} -- The Box folder to traverse.
	"""
	print("")
	print("File Listing")
	print(f"{'ID'.ljust(12)} Path")

	walk_folder_tree(folder, download_root)


## Main ##

if __name__ == "__main__":

	# Get a client instance for the service account.
	client = get_authenticated_client(pathToConfigJson)

	# Print the name and login associated with the service account.
	print_user_info(client)

	# get folderId
	shared_item = client.get_shared_item(shared_link_url)
	print(f"Shared folder name: {shared_item.name}")
	folderId = shared_item.id
	
	# Print a file and folder listing
	download_root = Path(local_download_path)
	assert(download_root.exists())
	print_folder_tree(shared_item, download_root)
