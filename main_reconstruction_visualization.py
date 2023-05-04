from pathlib import Path
from hloc.utils.read_write_model import read_model, qvec2rotmat
from hloc.utils import viz_3d


outputs = Path('/home/leonardo/elective_project/reconstruction_output')

# TODO set the model folder (reconstruction_folder)
reconstruction_folder = 'output_001'
outputs = outputs / reconstruction_folder
sfm_dir = outputs / 'sfm'


cameras, images, points3D = read_model(sfm_dir)
# print(cameras, images, points3D)



# for point3D in points3D:
#     print(points3D[point3D])
#     break

for camera in cameras:
    print(cameras[camera])    
    break

exit()
# Image
# - id              : int
# - qvec            : np.array()
# - tvec            : np.array()
# - camera_id       : int
# - name            : str
# - xys             : np.array() 
# - point3D_ids     : np.array()

# 
for image in images:
    rotmat = qvec2rotmat(images[image].qvec)
    tvec = images[image].tvec 
    # print(images[image].id) 
    # print(images[image].tvec) 
    # print(images[image].qvec) 
    print(images[image])
    break
    
    # print(images[image].xys.shape)      
    # print(images[image].point3D_ids.shape)
    

# fig = viz_3d.init_figure()
# # viz_3d.plot_points(fig, po)
# viz_3d.plot_camera()
# viz_3d.plot_reconstruction(fig, model, color='rgba(255,0,0,0.5)', name="mapping")
# fig.show()

