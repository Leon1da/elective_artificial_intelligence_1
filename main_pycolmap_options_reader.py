import pycolmap

# print(pycolmap.ImageReaderOptions().camera_params)
# exit()
print("pycolmap.IncrementalMapperOptions()")
options = pycolmap.IncrementalMapperOptions().todict()
for k in options:
    print(k, options[k])


print()
print("pycolmap.ImageReaderOptions()")
options = pycolmap.ImageReaderOptions().todict()
for k in options:
    print(k, options[k])
    

print()
print("pycolmap.CameraMode")
print(pycolmap.CameraMode.__members__.keys())
    
# print(options)