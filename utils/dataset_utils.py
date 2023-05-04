def remove_extension(filename):
    return filename.split('.')[0] # remove extension (.jpg)

def path_to_filename(path):
    return path.split('/')[-1] # remove directory (/)
    
def path_to_id(path):
    filename = path_to_filename(path)
    filename = remove_extension(filename)
    filename = int(filename)
    return filename


    