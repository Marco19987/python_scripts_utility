import nvisii

interactive = False
width_ = 640
height_ = 480
fx = 610.59326171875
fy = 610.605712890625
cx = 317.7075500488281
cy = 238.1421356201172
focal_length = 0  
obj_to_load = "/cad_models/banana.obj" # absolute path to cad model
object_name = "banana"  # obj name
mesh_scale = 0.001

nvisii.initialize(headless=not interactive, verbose=True)
nvisii.disable_updates()
# nvisii.disable_denoiser()

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
    mesh=nvisii.mesh.create_from_file(object_name, obj_to_load),
    transform=nvisii.transform.create(object_name),
    material=nvisii.material.create(object_name)
)

obj_mesh.get_transform().set_parent(camera.get_transform())

nvisii.sample_pixel_area(
    x_sample_interval=(.5, .5),
    y_sample_interval=(.5, .5)
)



obj_mesh = nvisii.entity.get(self.object_name)

        x = self.estimated_pose.pose.position.x
        y = -self.estimated_pose.pose.position.y
        z = -self.estimated_pose.pose.position.z

        p = np.array([x, y, z])
        f = np.array([0, 0, self.focal_length])

        p_new = p + np.multiply(np.divide((f-p), LA.norm(p-f)), sigma)
        scale_obj = LA.norm(p_new-f)/LA.norm(p-f)

        obj_mesh.get_transform().set_position(p_new)

        rotation_flip = nvisii.angleAxis(-nvisii.pi(),nvisii.vec3(1,0,0)) * nvisii.quat(self.estimated_pose.pose.orientation.w,
                                                        self.estimated_pose.pose.orientation.x, self.estimated_pose.pose.orientation.y, self.estimated_pose.pose.orientation.z)
        obj_mesh.get_transform().set_rotation(rotation_flip)

        obj_mesh.get_transform().set_scale(nvisii.vec3(scale_obj*self.mesh_scale))
        
        
        self.virtual_depth_array = nvisii.render_data(
            width=int(self.width_),
            height=int(self.height_),
            start_frame=0,
            frame_count=1,
            bounce=int(0),
            options="depth"
        )

        self.virtual_depth_array = np.array(
            self.virtual_depth_array).reshape(self.height_, self.width_, 4)
        self.virtual_depth_array = np.flipud(self.virtual_depth_array)