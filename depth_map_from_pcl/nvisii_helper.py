
def initialize_nvisii(interactive, camera_intrinsics, object_name, obj_file_path):
    import nvisii
    nvisii.initialize(headless= not interactive, verbose=True)
    nvisii.disable_updates()
    # nvisii.disable_denoiser()

    fx,fy,cx,cy,width_,height_ = camera_intrinsics
    camera = nvisii.entity.create(
        name="camera",
        transform=nvisii.transform.create("camera"),
        camera=nvisii.camera.create_from_intrinsics(
            name="camera",
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width_,
            height=height_
        )
    )
    camera.get_transform().look_at(
        at=(0, 0, 0),
        up=(0, -1, -1),
        eye=(1, 1, 0)
    )
    nvisii.set_camera_entity(camera)

    obj_mesh = nvisii.entity.create(
        name=object_name,
        mesh=nvisii.mesh.create_from_file(object_name, obj_file_path),
        transform=nvisii.transform.create(object_name),
        material=nvisii.material.create(object_name)
    )

    obj_mesh.get_transform().set_parent(camera.get_transform())

    nvisii.sample_pixel_area(
        x_sample_interval=(.5, .5),
        y_sample_interval=(.5, .5))
        
    
    return 

def change_object_mesh(old_object_name, new_object_name,obj_file_path):
    import nvisii
    camera = nvisii.entity.get("camera")
   
    nvisii.entity.remove(old_object_name)
    
    obj_mesh = nvisii.entity.create(
        name=new_object_name,
        mesh=nvisii.mesh.create_from_file(new_object_name, obj_file_path),
        transform=nvisii.transform.create(new_object_name),
        material=nvisii.material.create(new_object_name)
    )

    obj_mesh.get_transform().set_parent(camera.get_transform())
    return 

def deinitialize_nvisii():
    import nvisii
    nvisii.deinitialize()
    return