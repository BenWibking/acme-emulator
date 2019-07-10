from boxsdk import JWTAuth, Client
from boxsdk.object.folder import Folder
from pprint import pprint
import os.path

## Configuration ##

# Set the path to your JWT app config JSON file here!
pathToConfigJson = "boxapi_config.json"

# Set the path to a folder you'd like to traverse here!
shared_link_url = "https://osu.box.com/v/acme-simulations-reduced"


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


def print_path(item):
    """Print the ID and path of a given Box file or folder."""
    item_id = item.id.rjust(12, ' ')
    parents = map(lambda p: p['name'], item.path_collection['entries'])
    path = f"{'/'.join(parents)}/{item.name}".strip('/')
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

    
def walk_folder_tree(folder):
    """Traverse a Box folder tree, performing the specified action on every file and folder.
    
    Arguments:
        folder {Folder} -- The Box folder to traverse.
    """

    subitems = list(get_subitems(folder))
    
    for file in filter(lambda i: i.type=="file", subitems):
        print_path(file)

    for subfolder in filter(lambda i: i.type=="folder", subitems):
        walk_folder_tree(subfolder)

        
def print_folder_tree(folder):
    """Print the contents of a Box folder tree
    
    Arguments:
        folder {Folder} -- The Box folder to traverse.
    """
    print("")
    print("File Listing")
    print(f"{'ID'.ljust(12)} Path")

    walk_folder_tree(folder)


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
    print_folder_tree(shared_item)
