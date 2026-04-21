bl_info = {
    "name": "Camera Array Tool",
    "blender": (3, 6, 4),
    "category": "Object",
    "version": (3, 2, 0),
    "author": "Olli Huttunen - ToppiNappi 2025",
    "description": "With this tool you are able to create Camera Arrays around your models and render images from multiple cameras.",
    "location": "View3D > N-panel > Cam Array",
    "warning": "",
    "wiki_url": "",
    "support": 'COMMUNITY',
    "tracker_url": "",
}

import bpy
import math
import mathutils
import os
import numpy as np
import collections
import random
import bmesh
import time
from mathutils import Color


# Get the directory of this script, ensuring compatibility across installations
addon_dir = os.path.dirname(os.path.abspath(__file__))
blend_file_path = os.path.join(addon_dir, "assets", "predefined_objects.blend")

print(f"Using blend file path: {blend_file_path}")  # Debug-tuloste polun varmistamiseksi

# --- Camera Array Tool's functions and operators ---

class GeneratePreMadeObjectOperator(bpy.types.Operator):
    bl_idname = "object.generate_pre_made_object"
    bl_label = "Generate pre-made object"
    bl_description = "Generates a selected pre-made object"

    def execute(self, context):
        scene = context.scene
        mytool = scene.my_tool
        selected_object = mytool.object_selection

        # Define the object names to link/import
        object_mapping = {
            'HALF_DOME': "HalfdomeIcosphere",
            'CYLINDER': "D-Cylinder",
            'INTERIOR_TOWER': 'InteriorTower',
            'MIN_ANGLE_26': 'MinAngle26',
            'MIN_ANGLE_17': 'MinAngle17'
        }

        if not os.path.exists(blend_file_path):
            self.report({'ERROR'}, f"Blend file not found: {blend_file_path}")
            return {'CANCELLED'}

        if selected_object in object_mapping:
            object_name = object_mapping[selected_object]

            # Load the object from the blend file
            with bpy.data.libraries.load(blend_file_path, link=False) as (data_from, data_to):
                if object_name in data_from.objects:
                    data_to.objects = [object_name]
                else:
                    self.report({'ERROR'}, f"Object '{object_name}' not found in blend file.")
                    return {'CANCELLED'}

            if data_to.objects:
                new_object = data_to.objects[0]
                bpy.context.collection.objects.link(new_object)
                
                # Set the new object as the active object
                bpy.context.view_layer.objects.active = new_object
                context.scene.my_tool.target_object = new_object

                self.report({'INFO'}, f"Successfully added pre-made object: {new_object.name}")
            else:
                self.report({'ERROR'}, f"Object '{object_name}' not found in blend file.")
                return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "Invalid object selection.")
            return {'CANCELLED'}

        return {'FINISHED'}


class CreateCamerasOperator(bpy.types.Operator):
    bl_idname = "object.create_cameras_faces"
    bl_label = "Create Cameras"
    bl_description = "Create cameras at face centers and point them based on placement setting"

    def execute(self, context):
        # Varmistetaan, että päivityskäsittelijä on rekisteröity
        if handler_update not in bpy.app.handlers.depsgraph_update_post:
            bpy.app.handlers.depsgraph_update_post.append(handler_update)
            print("Camera Array Tool: Depsgraph update handler registered.")

        # Varmistetaan, että my_tool-ominaisuus on rekisteröity
        if not hasattr(bpy.types.Scene, "my_tool"):
            bpy.types.Scene.my_tool = bpy.props.PointerProperty(type=MyProperties)
            print("Camera Array Tool: Scene properties registered.")

        selected_obj = bpy.context.active_object
        camera_placement = context.scene.my_tool.camera_placement

        if selected_obj and selected_obj.type == 'MESH':
            bpy.context.view_layer.objects.active = selected_obj
            for modifier in selected_obj.modifiers:
                bpy.ops.object.modifier_apply(modifier=modifier.name)

            context.scene.my_tool.target_object = selected_obj
            collection_name = "Camera Array"

            if collection_name not in bpy.data.collections:
                camera_collection = bpy.data.collections.new(collection_name)
                bpy.context.scene.collection.children.link(camera_collection)
            else:
                camera_collection = bpy.data.collections[collection_name]

            for i, face in enumerate(selected_obj.data.polygons):
                face_center = selected_obj.matrix_world @ face.center
                normal = selected_obj.matrix_world.to_3x3() @ face.normal

                if camera_placement in ('IN', 'BOTH'):
                    create_camera(face_center, -normal, camera_collection, selected_obj, i, "IN")

                if camera_placement in ('OUT', 'BOTH'):
                    create_camera(face_center, normal, camera_collection, selected_obj, i, "OUT")

            set_wireframe_display(selected_obj)
            self.report({'INFO'}, f"Cameras created for '{selected_obj.name}', and wireframe added.")
        else:
            self.report({'ERROR'}, "No valid mesh object selected")
            return {'CANCELLED'}

        return {'FINISHED'}


def create_camera(face_center, normal, camera_collection, selected_obj, index, direction):
    camera_data = bpy.data.cameras.new(name=f"{selected_obj.name}_ArrayCam_{direction}.{str(index + 1).zfill(3)}")
    camera_object = bpy.data.objects.new(name=camera_data.name, object_data=camera_data)
    camera_object.location = face_center
    camera_object.scale = (0.2, 0.2, 0.2)
    camera_object.rotation_mode = 'QUATERNION'
    
    if direction == "IN":
        camera_object.rotation_quaternion = normal.to_track_quat('Z', 'Y')
    else:
        camera_object.rotation_quaternion = (-normal).to_track_quat('Z', 'Y')
    
    camera_collection.objects.link(camera_object)
    if camera_object.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(camera_object)


def handler_update(scene, depsgraph):
    """Update the cameras when the target object is modified"""
    for update in depsgraph.updates:
        obj = update.id
        if isinstance(obj, bpy.types.Object) and obj.type == 'MESH':
            update_cameras_for_object(obj)

def update_cameras_for_object(obj):
    """Update cameras only for a specific object and keep them at the centers of the polygons"""
    if obj and obj.type == 'MESH':
        camera_collection = bpy.data.collections.get("Camera Array")
        if not camera_collection:
            return
        cameras_out = [cam for cam in camera_collection.objects if cam.name.startswith(f"{obj.name}_ArrayCam_OUT")]
        cameras_in = [cam for cam in camera_collection.objects if cam.name.startswith(f"{obj.name}_ArrayCam_IN")]
        for i, face in enumerate(obj.data.polygons):
            face_center = obj.matrix_world @ face.center
            normal = obj.matrix_world.to_3x3() @ face.normal
            if i < len(cameras_out):
                camera_out = cameras_out[i]
                camera_out.location = face_center
                direction_out = (-normal).normalized()
                camera_out.rotation_quaternion = direction_out.to_track_quat('Z', 'Y')
            if i < len(cameras_in):
                camera_in = cameras_in[i]
                camera_in.location = face_center
                direction_in = normal.normalized()
                camera_in.rotation_quaternion = direction_in.to_track_quat('Z', 'Y')

def set_wireframe_display(obj):
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'WIREFRAME'
    obj.hide_render = True  # Piilotetaan objekti renderöinnistä


def create_camera_animation():
    scene = bpy.context.scene
    camera_collection = bpy.data.collections.get("Camera Array")
    
    if not camera_collection:
        print("Error: Camera Array collection not found.")
        return
    
    cameras = [obj for obj in camera_collection.objects if obj.type == 'CAMERA']
    total_frames = len(cameras)
    
    if total_frames == 0:
        print("Error: No cameras found in Camera Array collection.")
        return
    
    # Luo uusi kamera animaatiota varten
    animated_camera = bpy.data.objects.new("Animated_Camera", bpy.data.cameras.new("Animated_Camera"))
    scene.collection.objects.link(animated_camera)
    scene.camera = animated_camera
    
    # Aseta kameran skaalaukseksi sama kuin array-kameroilla
    animated_camera.scale = (0.2, 0.2, 0.2)
    
    # Asetetaan animaation pituus kameramäärän mukaan
    scene.frame_start = 1
    scene.frame_end = total_frames
    
    # Asetetaan uuden kameran animaatio vastaamaan jokaista array-kameran sijaintia ja rotaatiota
    for frame, camera in enumerate(cameras, start=1):
        animated_camera.location = camera.matrix_world.translation
        animated_camera.rotation_mode = 'QUATERNION'
        animated_camera.rotation_quaternion = camera.matrix_world.to_quaternion()
        
        animated_camera.keyframe_insert(data_path="location", frame=frame)
        animated_camera.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        animated_camera.keyframe_insert(data_path="scale", frame=frame)
    
    print(f"Successfully created animated camera with {total_frames} frames.")
    

class CreateCamerasAnimationOperator(bpy.types.Operator):
    """
    Operator to create an animation where a single camera moves through the Camera Array positions.
    """
    bl_idname = "object.create_cameras_animation"
    bl_label = "Create Cameras as an Animation"
    bl_description = "Create a single animated camera that moves through the Camera Array positions."
    
    @classmethod
    def poll(cls, context):
        camera_collection = bpy.data.collections.get("Camera Array")
        return camera_collection and len(camera_collection.objects) > 0 and any(obj.type == 'CAMERA' for obj in camera_collection.objects)
    
    def execute(self, context):
        create_camera_animation()
        self.report({'INFO'}, "Animated camera created successfully.")
        return {'FINISHED'}



def check_existing_images(render_path):
    """
    Etsii hakemistosta kuvatiedostot ja palauttaa ne järjestettynä.
    """
    if not os.path.exists(render_path) or not os.path.isdir(render_path):
        return []
    
    valid_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.exr']
    images = sorted(
        [f for f in os.listdir(render_path) if os.path.splitext(f)[1].lower() in valid_extensions]
    )
    return images

class UpdateImageCountOperator(bpy.types.Operator):
    bl_idname = "object.update_image_count"
    bl_label = "Update Image Count"
    bl_description = "Updates the image count based on selected Render Path"

    def execute(self, context):
        render_path = bpy.path.abspath(context.scene.my_tool.render_path)
        if os.path.isdir(render_path):
            existing_images = check_existing_images(render_path)
            context.scene.my_tool.image_count = len(existing_images) if existing_images else 0
            for area in bpy.context.screen.areas:
                if area.type == 'PROPERTIES':
                    area.tag_redraw()
            self.report({'INFO'}, f"Found {context.scene.my_tool.image_count} existing images.")
        else:
            context.scene.my_tool.image_count = 0
            for area in bpy.context.screen.areas:
                if area.type == 'PROPERTIES':
                    area.tag_redraw()
            self.report({'ERROR'}, "Render Path is invalid or does not exist.")
        return {'FINISHED'}
    


class RenderCamerasOperator(bpy.types.Operator):
    bl_idname = "object.render_cameras"
    bl_label = "Render Cameras"
    bl_description = "Render all cameras in the Camera Array collection"

    def execute(self, context):
        camera_collection = bpy.data.collections.get("Camera Array")
        render_path = context.scene.my_tool.render_path

        if not camera_collection:
            self.report({'ERROR'}, "Camera Array collection not found.")
            return {'CANCELLED'}

        cameras = [cam for cam in camera_collection.objects if cam.type == 'CAMERA']
        camera_count = len(cameras)

        if camera_count == 0:
            self.report({'ERROR'}, "No cameras found in the Camera Array collection.")
            return {'CANCELLED'}

        self.report({'INFO'}, f"Starting rendering of {camera_count} cameras.")

        file_format = context.scene.render.image_settings.file_format.lower()
        file_extension = self.get_file_extension(file_format)

        if not file_extension:
            self.report({'ERROR'}, "Unsupported file format selected.")
            return {'CANCELLED'}

        for index, camera in enumerate(cameras):
            context.scene.camera = camera
            filename = f"{camera.name}.{file_extension}"
            full_path = bpy.path.abspath(f"{render_path}/{filename}")
            context.scene.render.filepath = full_path

            self.report({'INFO'}, f"Rendering camera {index + 1}/{camera_count}: {camera.name}")
            
            bpy.ops.render.render(write_still=True)

            if not os.path.exists(full_path):
                self.report({'ERROR'}, f"Failed to save: {filename}")
                return {'CANCELLED'}

        self.report({'INFO'}, "Rendering completed for all cameras.")
        return {'FINISHED'}

    def invoke(self, context, event):
        camera_collection = bpy.data.collections.get("Camera Array")
        if camera_collection:
            cameras = [obj for obj in camera_collection.objects if obj.type == 'CAMERA']
            self.camera_count = len(cameras)
        else:
            self.camera_count = 0

        wm = context.window_manager
        return wm.invoke_props_dialog(self, width=300)

    def draw(self, context):
        layout = self.layout
        layout.label(text=f"Ready to render all {self.camera_count} cameras?")
        layout.label(text="You can follow the rendering progress in the File Explorer.")
        layout.label(text="Rendering cannot be cancelled once started.")

    def get_file_extension(self, file_format):
        format_extensions = {
            "bmp": "bmp",
            "file_output": "exr",
            "jpeg": "jpg",
            "jp2": "jp2",
            "openexr": "exr",
            "png": "png",
            "radiance_hdr": "hdr",
            "targa": "tga",
            "targa_raw": "tga",
            "tiff": "tif",
            "avi_jpeg": "avi",
            "avi_raw": "avi",
            "ffmpeg": "mp4",
            "cineon": "cin",
            "dpx": "dpx",
        }
        return format_extensions.get(file_format, None)



# --- COLMAP Exporter functionality ---

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
}

CAMERA_MODEL_IDS = {model.model_id: model for model in CAMERA_MODELS}
CAMERA_MODEL_NAMES = {model.model_name: model for model in CAMERA_MODELS}

def qvec2rotmat(qvec):
    rotmat = np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3], 2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], 1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1], 1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]
    ])
    return rotmat



class ExportCamerasOperator(bpy.types.Operator):
    bl_idname = "export.cameras"
    bl_label = "Generate cameras.txt"
    bl_description = "Generates the cameras.txt file for COLMAP from the camera array"

    def execute(self, context):
        render_path = bpy.context.scene.my_tool.render_path

        if not render_path or not os.path.isdir(bpy.path.abspath(render_path)):
            self.report({'ERROR'}, "Render Path is invalid or does not exist!")
            return {'CANCELLED'}

        cameras_txt_path = os.path.join(bpy.path.abspath(render_path), "cameras.txt")
        width = bpy.context.scene.render.resolution_x
        height = bpy.context.scene.render.resolution_y

        with open(cameras_txt_path, 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")

            cameras = [obj for obj in bpy.context.scene.objects if obj.type == 'CAMERA']
            if not cameras:
                self.report({'ERROR'}, "No cameras found!")
                return {'CANCELLED'}

            for camera_id, camera in enumerate(cameras):
                focal_length = camera.data.lens
                sensor_size = camera.data.sensor_width
                fx = fy = (focal_length / sensor_size) * width
                cx = width / 2
                cy = height / 2

                # Oletetaan vääristymäkertoimien arvot nolliksi
                k1, k2, p1, p2 = 0, 0, 0, 0

                f.write(f"{camera_id + 1} OPENCV {width} {height} {fx:.6f} {fy:.6f} {cx:.6f} {cy:.6f} {k1} {k2} {p1} {p2}\n")

        self.report({'INFO'}, f"cameras.txt saved to {cameras_txt_path}")
        return {'FINISHED'}


class ExportImagesOperator(bpy.types.Operator):
    bl_idname = "export.images"
    bl_label = "Generate images.txt"
    bl_description = "Generates the images.txt file for COLMAP"

    def execute(self, context):
        render_path = bpy.path.abspath(context.scene.my_tool.render_path)
        images_txt_path = os.path.join(render_path, "images.txt")
        
        camera_collection = bpy.data.collections.get("Camera Array")
        if not camera_collection:
            self.report({'ERROR'}, "Camera Array collection not found.")
            return {'CANCELLED'}
        
        cameras = [obj for obj in camera_collection.objects if obj.type == 'CAMERA']
        image_files = context.scene.my_tool.get("image_filenames", [])

        if len(image_files) != len(cameras):
            self.report({'WARNING'}, "Number of images and cameras do not match. Using available data.")
        
        with open(images_txt_path, 'w') as f:
            f.write("# Image list with two lines per image\n")
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            
            rotation_matrix = mathutils.Matrix.Rotation(math.radians(90), 4, 'X')

            for img_id, (camera, img_name) in enumerate(zip(cameras, image_files), start=1):
                camera.matrix_world = rotation_matrix @ camera.matrix_world
                bpy.context.view_layer.update()

                cam_rot_orig = mathutils.Quaternion(camera.rotation_quaternion)
                cam_rot = mathutils.Quaternion((
                    cam_rot_orig.x,
                    cam_rot_orig.w,
                    cam_rot_orig.z,
                    -cam_rot_orig.y))
                qw, qx, qy, qz = cam_rot.w, cam_rot.x, cam_rot.y, cam_rot.z

                T = mathutils.Vector(camera.location)
                T1 = -(cam_rot.to_matrix() @ T)
                tx, ty, tz = T1

                f.write(f"{img_id} {qw} {qx} {qy} {qz} {tx:.6f} {ty:.6f} {tz:.6f} {img_id} {img_name}\n")
                f.write("0.0 0.0 0\n")
            
            reverse_rotation_matrix = mathutils.Matrix.Rotation(-math.radians(90), 4, 'X')
            for camera in cameras:
                camera.matrix_world = reverse_rotation_matrix @ camera.matrix_world
            bpy.context.view_layer.update()
        
        self.report({'INFO'}, "images.txt successfully generated using existing image names and camera positions.")
        return {'FINISHED'}

    
# --- ColorPointCloud ---


class ExportPointsOperator(bpy.types.Operator):
    bl_idname = "export.points"
    bl_label = "Generate points3D.txt"
    bl_description = "Generates the points3D.txt file for COLMAP from the current model"

    def execute(self, context):
        import random
        import math

        render_path = bpy.path.abspath(context.scene.my_tool.render_path)
        density = context.scene.my_tool.density
        colored_points = context.scene.my_tool.colored_points  

        if not render_path or not os.path.isdir(render_path):
            self.report({'ERROR'}, "Render Path is invalid or does not exist!")
            return {'CANCELLED'}

        selected_obj = context.view_layer.objects.active
        if not selected_obj or selected_obj.type != 'MESH':
            self.report({'ERROR'}, "Please select a mesh object first!")
            return {'CANCELLED'}

        depsgraph = context.evaluated_depsgraph_get()
        evaluated_mesh = selected_obj.evaluated_get(depsgraph).to_mesh()

        if len(evaluated_mesh.vertices) == 0:
            self.report({'ERROR'}, "No vertices found in the selected object!")
            return {'CANCELLED'}

        bm = bmesh.new()
        bm.from_mesh(evaluated_mesh)

        uv_layer = bm.loops.layers.uv.active
        has_uv = uv_layer is not None

        # **Tarkistetaan, onko kyseessä Geometry Nodes -generoitu pistepilvi**
        geometry_nodes_generated = len(bm.faces) == 0 and len(bm.verts) > 0

        if geometry_nodes_generated:
            self.report({'WARNING'}, "No faces found! Using vertices directly from Geometry Nodes.")

        # **Haetaan kaikki objektin materiaalit**
        material_slots = selected_obj.material_slots
        material_cache = {}  # Välimuisti materiaaliviitteille
        material_color_cache = {}  # Välimuisti Base Color -väreille
        texture_cache = {}  # Välimuisti materiaalien tekstuuridatalle

        def get_material_color(material):
            """Palauttaa materiaalin Base Color, jos tekstuuria ei ole."""
            if material and material.use_nodes:
                for node in material.node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        return node.inputs['Base Color'].default_value[:3]  # RGB ilman alpha
            return (1.0, 1.0, 1.0)  # Oletus valkoinen

        points = []
        world_matrix = selected_obj.matrix_world

        if geometry_nodes_generated:
            # **Geometry Nodes -pistepilvi – käytetään verteksiä suoraan**
            for vert in bm.verts:
                world_coord = world_matrix @ vert.co
                x, y, z = world_coord.x, -world_coord.z, world_coord.y

                # **Kaikki pisteet valkoisiksi**
                r, g, b = 255, 255, 255  

                points.append((x, y, z, r, g, b, 0.0, ""))

        else:
            # **Tavallinen mesh, käytetään UV-karttaa ja tekstuuria**
            for face in bm.faces:
                material_index = face.material_index

                # **Haetaan oikea materiaali**
                if material_index in material_cache:
                    material = material_cache[material_index]
                else:
                    material = material_slots[material_index].material if material_index < len(material_slots) else None
                    material_cache[material_index] = material  # Tallennetaan välimuistiin

                # **Haetaan tekstuuri vain kerran per materiaali**
                if material_index not in texture_cache:
                    texture_image = find_base_color_texture(material) if has_uv else None
                    if texture_image:
                        image_pixels = list(texture_image.pixels)
                        image_width, image_height = texture_image.size
                    else:
                        image_pixels = []
                        image_width, image_height = 0, 0
                    texture_cache[material_index] = (image_pixels, image_width, image_height)
                else:
                    image_pixels, image_width, image_height = texture_cache[material_index]

                # **Haetaan materiaalin Base Color, mutta käytetään vain, jos tekstuuria ei ole**
                if material_index not in material_color_cache:
                    material_color_cache[material_index] = get_material_color(material)

                base_color = material_color_cache[material_index]

                for loop in face.loops:
                    vert = loop.vert
                    world_coord = world_matrix @ vert.co

                    x, y, z = world_coord
                    new_x = x
                    new_y = -z
                    new_z = y

                    r, g, b = [int(c * 255) for c in base_color]  # **Oletus Base Color**

                    if colored_points and image_pixels:
                        uv = loop[uv_layer].uv
                        img_x = min(max(int(uv.x * image_width), 0), image_width - 1)
                        img_y = min(max(int(uv.y * image_height), 0), image_height - 1)
                        pixel_index = (img_y * image_width + img_x) * 4

                        if pixel_index + 2 < len(image_pixels):
                            r = int(image_pixels[pixel_index] * 255)
                            g = int(image_pixels[pixel_index + 1] * 255)
                            b = int(image_pixels[pixel_index + 2] * 255)

                    points.append((new_x, new_y, new_z, r, g, b, 0.0, ""))

        if not points:
            self.report({'ERROR'}, "No points were generated!")
            bm.free()
            return {'CANCELLED'}
        
        # **Laske kuinka monta pistettä säilytetään density-arvon perusteella**
        total_points = len(points)
        num_points_to_keep = max(1, int((density / 100) * total_points))

        # **Vähennetään pisteitä, jos density ei ole 100%**
        if num_points_to_keep < total_points:
            points = random.sample(points, num_points_to_keep)

        points_txt_path = os.path.join(render_path, "points3D.txt")

        with open(points_txt_path, 'w') as f:
            f.write("# 3D point list with one line per point\n")
            f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")

            for point_id, (x, y, z, r, g, b, error, track_data) in enumerate(points):
                f.write(f"{point_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {error:.6f} {track_data}\n")

        bm.free()
        selected_obj.evaluated_get(depsgraph).to_mesh_clear()

        self.report({'INFO'}, f"points3D.txt saved to {points_txt_path} with {len(points)} points.")
        return {'FINISHED'}




# Funktio, joka etsii Base Color -tekstuurin shader-puusta
def find_base_color_texture(material):
    if not material or not material.use_nodes:
        return None

    for node in material.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            base_color_input = node.inputs['Base Color']
            if base_color_input.is_linked:
                linked_node = base_color_input.links[0].from_node
                if linked_node.type == 'TEX_IMAGE':
                    return linked_node.image
    return None


# --- Connected button function for Cameras and Images txt files ---

class ExportSceneAgentPosesOperator(bpy.types.Operator):
    """Exports the active Camera Array as the JSON file SceneAgent's
    pipeline expects (pipeline/output/camera_poses.json).

    Schema (one entry per camera):
      {index, position[3], look_at[3], up[3], fov_y_deg, width, height,
       cluster_size}

    Coordinate system: Blender world coords (Z-up, right-handed). The
    splat must be at world origin with no rotation when you export, or
    the cameras will be in a different frame than the splat the
    pipeline rasterises.
    """
    bl_idname = "export.sceneagent_poses"
    bl_label = "Export to SceneAgent (camera_poses.json)"
    bl_description = (
        "Walk the 'Camera Array' collection and write a JSON file the "
        "SceneAgent pipeline can use as camera_poses.json"
    )

    filepath: bpy.props.StringProperty(
        name="File Path",
        description="Where to write camera_poses.json",
        default="camera_poses.json",
        subtype='FILE_PATH',
    )
    filename_ext = ".json"
    filter_glob: bpy.props.StringProperty(default="*.json", options={'HIDDEN'})

    def invoke(self, context, event):
        # Default to the addon-side render path if set, otherwise let Blender
        # pop the standard file browser at home.
        rp = context.scene.my_tool.render_path
        if rp:
            self.filepath = os.path.join(bpy.path.abspath(rp), "camera_poses.json")
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        camera_collection = bpy.data.collections.get("Camera Array")
        if camera_collection is None:
            self.report({'ERROR'}, "No 'Camera Array' collection found. "
                        "Run 'Create Cameras' first.")
            return {'CANCELLED'}

        cameras = [o for o in camera_collection.objects if o.type == 'CAMERA']
        if not cameras:
            self.report({'ERROR'}, "Camera Array collection has no cameras.")
            return {'CANCELLED'}

        # sort by name so the index order is reproducible across Blender sessions
        cameras.sort(key=lambda c: c.name)

        scene = context.scene
        width = int(scene.render.resolution_x)
        height = int(scene.render.resolution_y)

        out = []
        for i, cam in enumerate(cameras):
            mw = cam.matrix_world
            pos = mw.translation
            # Blender camera looks down its local -Z axis. Convert to a
            # look_at world point along that direction (1 unit forward).
            forward = (mw.to_3x3() @ mathutils.Vector((0.0, 0.0, -1.0))).normalized()
            look_at = pos + forward

            # vertical FoV in degrees. cam.data.angle_y is radians and
            # respects sensor_fit + lens; matches what scene.render does.
            fov_y_deg = math.degrees(cam.data.angle_y)

            out.append({
                "index": i,
                "name": cam.name,
                "position": [float(pos.x), float(pos.y), float(pos.z)],
                "look_at": [float(look_at.x), float(look_at.y), float(look_at.z)],
                "up": [0.0, 0.0, 1.0],
                "fov_y_deg": float(fov_y_deg),
                "width": width,
                "height": height,
                "cluster_size": 1,
            })

        path = bpy.path.abspath(self.filepath)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        import json as _json
        with open(path, "w") as f:
            _json.dump(out, f, indent=2)

        self.report({'INFO'},
                    f"Wrote {len(out)} poses to {path}")
        return {'FINISHED'}


class ExportCamerasAndImagesOperator(bpy.types.Operator):
    bl_idname = "export.cameras_and_images"
    bl_label = "Generate Cameras and Images txt files"
    bl_description = "Generates cameras.txt and images.txt files for COLMAP based on existing images and Camera Array"

    def execute(self, context):
        render_path = bpy.path.abspath(context.scene.my_tool.render_path)
        if not render_path or not os.path.isdir(render_path):
            self.report({'ERROR'}, "Invalid render path! Cannot generate COLMAP files.")
            return {'CANCELLED'}
        
        image_files = check_existing_images(render_path)
        if not image_files:
            self.report({'ERROR'}, "No images found in the selected directory! Cannot generate COLMAP files.")
            return {'CANCELLED'}
        
        # Tallennetaan kuvien nimet Blenderin scene-datarakenteeseen
        context.scene.my_tool["image_filenames"] = image_files
        
        # Käytetään olemassa olevia operaatioita COLMAP-tiedostojen generointiin
        bpy.ops.export.cameras()
        bpy.ops.export.images()
        
        self.report({'INFO'}, "Cameras and Images txt files successfully generated using existing images.")
        return {'FINISHED'}


# --- Merge Objects ---

class MergeObjectsOperator(bpy.types.Operator):
    bl_idname = "object.merge_objects"
    bl_label = "Merge objects for points3D data"
    bl_description = "Merges selected objects, including Parent objects and their children, while preserving Shape Keys and Vertex Groups"

    def execute(self, context):
        selected_objects = bpy.context.selected_objects
        total_objects = len(selected_objects)

        if total_objects < 2:
            self.report({'ERROR'}, "You need at least two objects to merge!")
            return {'CANCELLED'}

        # **Etsitään kaikki MESH-objektit, mukaan lukien Parent-objektien lapset**
        all_mesh_objects = set()
        for obj in selected_objects:
            if obj.type == 'MESH':
                all_mesh_objects.add(obj)
            if obj.children:
                for child in obj.children_recursive:
                    if child.type == 'MESH':
                        all_mesh_objects.add(child)

        total_mesh_objects = len(all_mesh_objects)

        if total_mesh_objects < 2:
            self.report({'ERROR'}, f"Only {total_mesh_objects} MESH object(s) found. At least two are required!")
            return {'CANCELLED'}
        
        self.report({'INFO'}, f"Found {total_mesh_objects} MESH objects (including children). Proceeding with merge...")

        # **Asetetaan aktiivinen objekti, jos sitä ei ole**
        if bpy.context.view_layer.objects.active is None:
            bpy.context.view_layer.objects.active = list(all_mesh_objects)[0]

        # **Varmistetaan, että aktiivinen objekti on MESH ennen OBJECT-tilaan siirtymistä**
        if bpy.context.object and bpy.context.object.type == 'MESH':
            bpy.ops.object.mode_set(mode='OBJECT')

        # **Duplikoidaan vain MESH-objektit**
        bpy.ops.object.select_all(action='DESELECT')
        for obj in all_mesh_objects:
            obj.select_set(True)
        bpy.ops.object.duplicate()
        duplicated_objects = [obj for obj in bpy.context.selected_objects if obj.select_get()]

        self.report({'INFO'}, f"Duplicated {len(duplicated_objects)} objects for processing.")

        # **Apply viimeisin Shape Key manuaalisesti**
        shape_keys_applied = 0
        for obj in duplicated_objects:
            if obj.type == 'MESH' and obj.data.shape_keys is not None:
                shape_keys = obj.data.shape_keys.key_blocks
                if len(shape_keys) > 1:
                    last_active_key = shape_keys[-1]
                    
                    # **Soveltaa Shape Keyn suoraan**
                    for i, vert in enumerate(obj.data.vertices):
                        vert.co = last_active_key.data[i].co.copy()
                    shape_keys_applied += 1

        if shape_keys_applied > 0:
            self.report({'INFO'}, f"Applied Shape Keys to {shape_keys_applied} objects.")

        # **Poistetaan kaikki Shape Keys**
        for obj in duplicated_objects:
            if obj.type == 'MESH' and obj.data.shape_keys is not None:
                bpy.context.view_layer.objects.active = obj
                while obj.data.shape_keys and len(obj.data.shape_keys.key_blocks) > 0:
                    bpy.ops.object.shape_key_remove(all=True)

        # **Apply modifierit**
        modifiers_applied = 0
        for obj in duplicated_objects:
            bpy.context.view_layer.objects.active = obj
            for modifier in obj.modifiers:
                try:
                    bpy.ops.object.modifier_apply(modifier=modifier.name)
                    modifiers_applied += 1
                except RuntimeError:
                    self.report({'WARNING'}, f"Could not apply modifier {modifier.name} on {obj.name}")

        if modifiers_applied > 0:
            self.report({'INFO'}, f"Applied {modifiers_applied} modifiers.")

        # **Säilytetään Vertex Groupit**
        for obj in duplicated_objects:
            if obj.type == 'MESH' and obj.vertex_groups:
                for group in obj.vertex_groups:
                    group.name = f"{obj.name}_{group.name}"

        # **Varmistetaan, että yhdistettävät objektit ovat valittuina**
        bpy.ops.object.select_all(action='DESELECT')
        for obj in duplicated_objects:
            obj.select_set(True)

        # **Asetetaan aktiiviseksi varmistettu MESH-objekti**
        bpy.context.view_layer.objects.active = duplicated_objects[0]

        # **Yritetään yhdistää objektit**
        try:
            bpy.ops.object.join()
        except RuntimeError:
            self.report({'ERROR'}, "Failed to merge objects. Make sure all selected objects are MESH type.")
            return {'CANCELLED'}

        # **Varmistetaan, että yhdistetty objekti on olemassa**
        merged_object = bpy.context.active_object
        if merged_object is None or merged_object.type != 'MESH':
            self.report({'ERROR'}, "Merging failed, no valid object was created!")
            return {'CANCELLED'}

        # **Varmistetaan, että uusi objekti siirretään Outlinerin juuritasolle**
        if merged_object.parent:
            merged_object.parent = None  # Poistetaan Parent-suhde
            self.report({'INFO'}, f"Removed parent relationship from {merged_object.name}.")

        # **Asetetaan Outlinerissa objekti näkyväksi ja valituksi**
        merged_object.select_set(True)
        bpy.context.view_layer.objects.active = merged_object

        # **Nimetään yhdistetty objekti**
        merged_object.name = "Merged_point3D_mesh"
        merged_object.hide_render = False
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        self.report({'INFO'}, f"Successfully merged {total_mesh_objects} objects into {merged_object.name}. It is now at the root level in the Outliner.")
        return {'FINISHED'}


# --- Pointcloud with Geometry nodes ---
  
class AddGeometryNodesPointCloudOperator(bpy.types.Operator):
    bl_idname = "object.add_geometry_nodes_point_cloud"
    bl_label = "Add GeometryNodes Point Cloud"

    def execute(self, context):
        blend_file = os.path.join(addon_dir, "assets", "predefined_objects.blend")
        geometry_nodes_name = "PointCloudGenerator"  
        selected_obj = bpy.context.active_object

        if selected_obj and selected_obj.type == 'MESH':
            if not os.path.exists(blend_file):
                self.report({'ERROR'}, f"Blend file not found: {blend_file}")
                return {'CANCELLED'}

            bpy.ops.wm.append(
                filepath=os.path.join(blend_file, "NodeTree", geometry_nodes_name),
                directory=os.path.join(blend_file, "NodeTree"),
                filename=geometry_nodes_name
            )
            
            geometry_nodes_tree = bpy.data.node_groups.get(geometry_nodes_name)
            if geometry_nodes_tree:
                gn_modifier = selected_obj.modifiers.new(name="PointCloudGenerator", type='NODES')
                gn_modifier.node_group = geometry_nodes_tree
                self.report({'INFO'}, "Geometry Nodes Point Cloud added to the selected object.")
            else:
                self.report({'ERROR'}, "Failed to load the Geometry Nodes tree.")
                return {'CANCELLED'}
        else:
            self.report({'ERROR'}, "No valid mesh object selected.")
            return {'CANCELLED'}

        return {'FINISHED'}
 
 
  # --- 4DGS ---   
    

class Render4DGSAnimationOperator(bpy.types.Operator):
    bl_idname = "object.render_4dgs_animation"
    bl_label = "Render 4DGS Animation"
    bl_description = "Render animation frames for 4D Gaussian Splatting"

    def execute(self, context):
        scene = context.scene
        mytool = scene.my_tool
        animated_obj = mytool.animated_object

        if mytool.include_colmap_data and not animated_obj:
            self.report({'ERROR'}, "No animated object selected. Please select an object to include COLMAP data.")
            return {'CANCELLED'}

        print("Starting 4DGS animation render...")
        frame_start = int(scene.frame_start)
        frame_end = int(scene.frame_end)
        step = int(scene.frame_step)
        base_render_path = bpy.path.abspath(scene.my_tool.render_path)

        if not base_render_path:
            self.report({'ERROR'}, "Render path is not set.")
            return {'CANCELLED'}
        if not os.path.exists(base_render_path):
            try:
                os.makedirs(base_render_path)
            except Exception as e:
                self.report({'ERROR'}, f"Failed to create render path: {e}")
                return {'CANCELLED'}

        camera_collection = bpy.data.collections.get("Camera Array")
        if not camera_collection:
            self.report({'ERROR'}, "Camera Array collection not found.")
            return {'CANCELLED'}

        for frame_number in range(frame_start, frame_end + 1, step):
            frame_folder = os.path.join(base_render_path, f"Frame{frame_number:04d}")

            already_rendered = False
            if mytool.resume_rendering and os.path.exists(frame_folder):
                existing_files = os.listdir(frame_folder)
                file_format = scene.render.image_settings.file_format.lower()
                file_extension = {
                    'png': '.png',
                    'jpeg': '.jpg',
                    'tiff': '.tiff',
                    'bmp': '.bmp',
                    'openexr': '.exr'
                }.get(file_format, '.png')

                if any(f.endswith(file_extension) for f in existing_files):
                    print(f"Skipping frame {frame_number}: already rendered in {frame_folder}")
                    already_rendered = True

            if already_rendered:
                continue

            print(f"\nRendering frame {frame_number} to {frame_folder}")
            render_frame_with_colmap(context, scene, camera_collection, frame_number, frame_folder)

        self.report({'INFO'}, "4DGS Animation rendering completed successfully.")
        return {'FINISHED'}

    def invoke(self, context, event):
        scene = context.scene
        mytool = scene.my_tool

        if mytool.include_colmap_data and not mytool.animated_object:
            self.report({'ERROR'}, "No animated object selected. Please select an object to include COLMAP data.")
            return {'CANCELLED'}

        camera_collection = bpy.data.collections.get("Camera Array")
        if camera_collection:
            cameras = [obj for obj in camera_collection.objects if obj.type == 'CAMERA']
            self.camera_count = len(cameras)
        else:
            self.camera_count = 0

        self.frame_start = scene.frame_start
        self.frame_end = scene.frame_end
        self.frame_step = scene.frame_step
        self.frame_count = len(range(self.frame_start, self.frame_end + 1, self.frame_step))

        # Laske renderöitävät framet
        resume_enabled = mytool.resume_rendering
        base_render_path = bpy.path.abspath(scene.my_tool.render_path)
        already_done = 0

        if resume_enabled and os.path.exists(base_render_path):
            for frame_number in range(self.frame_start, self.frame_end + 1, self.frame_step):
                frame_folder = os.path.join(base_render_path, f"Frame{frame_number:04d}")
                if os.path.exists(frame_folder):
                    existing_files = os.listdir(frame_folder)
                    file_format = scene.render.image_settings.file_format.lower()
                    file_extension = {
                        'png': '.png',
                        'jpeg': '.jpg',
                        'tiff': '.tiff',
                        'bmp': '.bmp',
                        'openexr': '.exr'
                    }.get(file_format, '.png')
                    if any(f.endswith(file_extension) for f in existing_files):
                        already_done += 1

        self.frames_to_render = self.frame_count if not resume_enabled else self.frame_count - already_done
        self.already_rendered = already_done
        self.resume_enabled = resume_enabled

        wm = context.window_manager
        return wm.invoke_props_dialog(self, width=420)

    def draw(self, context):
        layout = self.layout
        if self.resume_enabled:
            layout.label(text=f"Resuming: {self.already_rendered} frames already rendered.")
            layout.label(text=f"{self.frames_to_render} frames remaining x {self.camera_count} cameras = {self.frames_to_render * self.camera_count} renders.")
        else:
            layout.label(text=f"Rendering full animation: {self.frame_count} frames x {self.camera_count} cameras = {self.frame_count * self.camera_count} renders.")

        layout.label(text="You can follow the rendering progress in the File Explorer.")
        layout.label(text="Rendering cannot be cancelled once started.")




def render_images(scene, base_render_path, frame_start, frame_end, camera_collection):
    frame_data = []  # Tallentaa hakemiston polun ja frame-numeron

    # Hae Step-arvo Blenderin asetuksista
    step = scene.frame_step

    # Tiedostomuodon selvittäminen
    file_format = scene.render.image_settings.file_format.lower()  # PNG, JPEG, etc.
    file_extension = {
        'png': '.png',
        'jpeg': '.jpg',
        'tiff': '.tiff',
        'bmp': '.bmp',
        'openexr': '.exr'
    }.get(file_format, '.png')  # Oletuksena .png, jos formaatti ei tunnisteta

    for current_frame in range(frame_start, frame_end + 1, step):
        scene.frame_set(current_frame)

        # Luo hakemisto tälle framelle
        frame_folder = os.path.join(base_render_path, f"Frame{current_frame:04d}")
        try:
            os.makedirs(frame_folder, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create directory {frame_folder}: {e}")
        
        frame_data.append((frame_folder, current_frame))  # Tallenna polku ja frame-numero

        # Renderöi kuvat kaikille kameroille
        for camera in camera_collection.objects:
            if camera.type == 'CAMERA':
                scene.camera = camera

                # Siisti tiedostonimi mahdollisista erikoismerkeistä
                valid_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in camera.name)
                image_filename = f"{valid_name}{file_extension}"
                full_image_path = os.path.join(frame_folder, image_filename)

                # Aseta renderöintipolku
                scene.render.filepath = full_image_path

                try:
                    bpy.ops.render.render(write_still=True)
                except Exception as e:
                    raise RuntimeError(f"Render failed for camera {camera.name} at frame {current_frame}: {e}")

                # Varmista, että kuva tallennettiin
                if not os.path.exists(full_image_path):
                    raise RuntimeError(f"Failed to save rendered image: {full_image_path}. File not found after render.")

    return frame_data


def generate_colmap_data(scene, frame_data):
    animated_obj = scene.my_tool.animated_object

    if not animated_obj or animated_obj.type != 'MESH':
        print("Error: No valid animated object selected!")
        return

    for frame_folder, frame_number in frame_data:
        scene.frame_set(frame_number)

        bpy.ops.object.select_all(action='DESELECT')
        bpy.context.view_layer.objects.active = animated_obj
        animated_obj.select_set(True)

        # Aseta kuvatiedostot COLMAP-exporttia varten
        image_files = sorted([
            f for f in os.listdir(frame_folder)
            if os.path.splitext(f)[1].lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.exr']
        ])
        scene.my_tool["image_filenames"] = image_files

        # Päivitä renderöintipolku tilapäisesti
        original_render_path = scene.my_tool.render_path
        scene.my_tool.render_path = frame_folder

        bpy.ops.export.cameras()
        bpy.ops.export.images()
        bpy.ops.export.points()

        scene.my_tool.render_path = original_render_path

    print(f"COLMAP data generated successfully in {len(frame_data)} frames.")



def render_frame_with_colmap(context, scene, camera_collection, frame_number, frame_folder):
    scene.frame_set(frame_number)

    try:
        os.makedirs(frame_folder, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create directory {frame_folder}: {e}")

    file_format = scene.render.image_settings.file_format.lower()
    file_extension = {
        'png': '.png',
        'jpeg': '.jpg',
        'tiff': '.tiff',
        'bmp': '.bmp',
        'openexr': '.exr'
    }.get(file_format, '.png')

    image_filenames = []

    for camera in camera_collection.objects:
        if camera.type == 'CAMERA':
            scene.camera = camera
            # Tee validi nimi, kuten ExportImagesOperator odottaa
            valid_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in camera.name)
            image_filename = f"{valid_name}{file_extension}"
            image_filenames.append(image_filename)
            image_path = os.path.join(frame_folder, image_filename)
            scene.render.filepath = image_path

            try:
                bpy.ops.render.render(write_still=True)
            except Exception as e:
                raise RuntimeError(f"Render failed for camera {camera.name} at frame {frame_number}: {e}")

            if not os.path.exists(image_path):
                raise RuntimeError(f"Failed to save rendered image: {image_path}")

    # Aseta render_path ja image_filenames COLMAP exporttereita varten
    original_render_path = scene.my_tool.render_path
    scene.my_tool.render_path = frame_folder
    scene.my_tool["image_filenames"] = sorted(image_filenames)

    # COLMAP exportit
    bpy.ops.export.cameras()
    bpy.ops.export.images()

    animated_obj = scene.my_tool.animated_object
    if animated_obj and animated_obj.type == 'MESH':
        bpy.ops.object.select_all(action='DESELECT')
        animated_obj.select_set(True)
        scene.view_layers[0].objects.active = animated_obj
    else:
        raise RuntimeError("No animated mesh object selected for COLMAP point export.")

    bpy.ops.export.points()

    # Palauta alkuperäinen render_path
    scene.my_tool.render_path = original_render_path



# --- MyProperties ---

class MyProperties(bpy.types.PropertyGroup):
    render_path: bpy.props.StringProperty(
        name="Render Path",
        description="Directory to save rendered images and Colmap files",
        default="//",
        subtype='DIR_PATH',
        update=lambda self, context: bpy.ops.object.update_image_count()
    )
    image_count: bpy.props.IntProperty(
        name="Existing Images",
        description="Number of existing images in the Render Path",
        default=0
    )
    camera_placement: bpy.props.EnumProperty(
        name="Camera Placement",
        description="Choose camera placement: Inward, Outward, or Both",
        items=[
            ('IN', "Inward", "Cameras point inward"),
            ('OUT', "Outward", "Cameras point outward"),
            ('BOTH', "Both", "Cameras point both directions")
        ],
        default='IN'
    )
    focal_length: bpy.props.FloatProperty(
        name="Focal Length",
        description="Focal length for all cameras",
        default=35.0,
        min=1.0,
        max=300.0,
        update=lambda self, context: update_focal_length(context)
    )
    target_object: bpy.props.PointerProperty(
        name="Target Object",
        description="Object to track for camera updates",
        type=bpy.types.Object
    )
    show_advanced: bpy.props.BoolProperty(
        name="COLMAP Exporter",
        description="Expand to show COLMAP Exporter settings",
        default=False
    )
    object_selection: bpy.props.EnumProperty(
        name="Choose object",
        description="Choose pre-made object from list",
        items=[
            ('HALF_DOME', "Half Dome", "Half Dome shape"),
            ('CYLINDER', "Cylinder", "Cylinder shape"),
            ('INTERIOR_TOWER', "Interior Tower", "Camera Tower shape for interiors"),
            ('MIN_ANGLE_26', "Minimum Cam Angles 26", "Minimum Cam Angles 26"),
            ('MIN_ANGLE_17', "Minimum Cam Angles 17", "Minimum Cam Angles 17")
        ],
        default='HALF_DOME'
    )
    density: bpy.props.IntProperty(
        name="Density",
        description="Percentage of points to keep (100 = all points)",
        default=100,
        min=1,
        max=100
    )
    colored_points: bpy.props.BoolProperty(
        name="Colored Points",
        description="Generate point cloud with colors. If unchecked, points will be white.",
        default=True  # Oletuksena päällä
    )
    animated_object: bpy.props.PointerProperty(
        name="Animated Object",
        description="Select the animated object for 4DGS rendering",
        type=bpy.types.Object
    )
    show_animated_4dgs: bpy.props.BoolProperty(
        name="Animated 4DGS Data",
        description="Expand to show Animated 4DGS settings",
        default=False
    )
    include_colmap_data: bpy.props.BoolProperty(
        name="Include COLMAP Data",
        description="Render 4DGS Animation with COLMAP data",
        default=True
    )
    show_additional_tools: bpy.props.BoolProperty(
        name="Additional Tools",
        description="Show or hide additional tools",
        default=False  # Oletusarvoisesti kiinni
    )
    resume_rendering: bpy.props.BoolProperty(
        name="Resume Rendering",
        description="Skip frames that have already been rendered",
        default=True
    )




# --- UI ---

class CameraArrayPanel(bpy.types.Panel):
    bl_label = f"Camera Array Tool v{'.'.join(map(str, bl_info['version']))}"
    bl_idname = "VIEW3D_PT_camera_array_tool"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Cam Array"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        mytool = scene.my_tool
        
        layout.label(text="Pre made array objects", icon='CUBE')
        layout.prop(mytool, "object_selection")
        layout.operator("object.generate_pre_made_object", text="Generate pre-made object")

        layout.label(text="Camera Placement", icon='URL')
        layout.prop(mytool, "camera_placement")
        layout.label(text="Create Cameras")
        layout.prop(mytool, "target_object")
        layout.operator("object.create_cameras_faces")
        layout.operator("object.create_cameras_animation", text="Create Cameras as an Animation")
        layout.separator()
        layout.prop(mytool, "focal_length")
        layout.prop(mytool, "render_path")
        layout.label(text=f"Existing images: {mytool.image_count}", icon='IMAGE')
        layout.label(text="Render Cameras", icon='OUTLINER_OB_CAMERA')
        layout.operator("object.render_cameras")

        # SceneAgent integration: one-button export of camera array →
        # camera_poses.json that the SceneAgent pipeline understands.
        sa_box = layout.box()
        sa_box.label(text="SceneAgent Export", icon='WORLD_DATA')
        sa_box.operator("export.sceneagent_poses",
                        text="Export to SceneAgent (camera_poses.json)")

        layout.label(text="3D Data features")
        
        
            
        
        layout.prop(mytool, "show_advanced", icon="TRIA_DOWN" if mytool.show_advanced else "TRIA_RIGHT", emboss=True)
        if mytool.show_advanced:
            colmap_box = layout.box()
            colmap_box.label(text="COLMAP Data Export", icon='EXPORT')
            colmap_box.operator("export.cameras_and_images", text="Generate Cameras and Images txt files")
            colmap_box.label(text="Point Cloud Settings", icon='MOD_PARTICLES')
            colmap_box.prop(mytool, "density")
            colmap_box.prop(mytool, "colored_points", text="Colored Points")
            colmap_box.operator("export.points", text="Generate points3D.txt")
            colmap_box.label(text="Advanced")

            # Additional Tools collapsible section
            colmap_box.prop(mytool, "show_additional_tools", icon="TRIA_DOWN" if mytool.show_additional_tools else "TRIA_RIGHT", emboss=True)
            if mytool.show_additional_tools:
                additional_box = colmap_box.box()
                additional_box.label(text="Select objects and merge for points3D data", icon='INFO')
                additional_box.operator("object.merge_objects", text="Merge objects")
                additional_box.label(text="Generate Geometry Node Point Cloud to object", icon='STICKY_UVS_DISABLE')
                additional_box.operator("object.add_geometry_nodes_point_cloud", text="Add Random Point Cloud modifier")

        layout.prop(mytool, "show_animated_4dgs", icon="TRIA_DOWN" if mytool.show_animated_4dgs else "TRIA_RIGHT", emboss=True)
        if mytool.show_animated_4dgs:
            animated_4dgs_box = layout.box()
            animated_4dgs_box.label(text="Animated Object", icon='OUTLINER_OB_ARMATURE')
            animated_4dgs_box.prop(mytool, "animated_object")
            animated_4dgs_box.prop(mytool, "include_colmap_data")
            animated_4dgs_box.prop(mytool, "resume_rendering", text="Resume from last rendered frame")
            animated_4dgs_box.operator("object.render_4dgs_animation", text="Render 4DGS Animation")

        
    
# --- Helper Functions ---

def set_wireframe_display(obj):
    obj.display_type = 'WIRE'  
    obj.show_all_edges = True
    obj.hide_render = True

def update_focal_length(context):
    focal_length = context.scene.my_tool.focal_length
    for camera in bpy.data.objects:
        if camera.type == 'CAMERA':
            camera.data.lens = focal_length
    print(f"Focal length updated to {focal_length} mm for all cameras")

def on_new_file(scene):
    if handler_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(handler_update)
        print("Camera Array Tool: Depsgraph update handler re-registered after new file.")
    else:
        print("Camera Array Tool: Depsgraph handler already registered.")

def ensure_tool_properties():
    if "my_tool" not in bpy.types.Scene.bl_rna.properties:
        bpy.types.Scene.my_tool = bpy.props.PointerProperty(type=MyProperties)
        print("Camera Array Tool: Scene properties registered.")
    else:
        print("Camera Array Tool: Scene properties already registered.")


# --- Registration ---

def register():
    bpy.utils.register_class(MyProperties)
    ensure_tool_properties()

    bpy.utils.register_class(CreateCamerasOperator)
    bpy.utils.register_class(CreateCamerasAnimationOperator)
    bpy.utils.register_class(UpdateImageCountOperator)
    bpy.utils.register_class(RenderCamerasOperator)
    bpy.utils.register_class(ExportCamerasOperator)
    bpy.utils.register_class(ExportImagesOperator)
    bpy.utils.register_class(ExportPointsOperator)
    bpy.utils.register_class(MergeObjectsOperator)
    bpy.utils.register_class(CameraArrayPanel)
    bpy.utils.register_class(ExportCamerasAndImagesOperator)
    bpy.utils.register_class(ExportSceneAgentPosesOperator)
    bpy.utils.register_class(GeneratePreMadeObjectOperator)
    bpy.utils.register_class(AddGeometryNodesPointCloudOperator)
    bpy.utils.register_class(Render4DGSAnimationOperator)
    bpy.types.Scene.my_tool = bpy.props.PointerProperty(type=MyProperties)

    if on_new_file not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(on_new_file)
        print("Camera Array Tool: Load handler registered.")

    if handler_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(handler_update)
        print("Camera Array Tool: Depsgraph handler registered.")

def unregister():
    if "my_tool" in bpy.types.Scene.bl_rna.properties:
        del bpy.types.Scene.my_tool
        print("Camera Array Tool: Scene properties unregistered.")

    bpy.utils.unregister_class(MyProperties)
    bpy.utils.unregister_class(CreateCamerasOperator)
    bpy.utils.unregister_class(CreateCamerasAnimationOperator)
    bpy.utils.unregister_class(UpdateImageCountOperator)
    bpy.utils.unregister_class(RenderCamerasOperator)
    bpy.utils.unregister_class(ExportCamerasOperator)
    bpy.utils.unregister_class(ExportImagesOperator)
    bpy.utils.unregister_class(ExportPointsOperator)
    bpy.utils.unregister_class(MergeObjectsOperator)
    bpy.utils.unregister_class(CameraArrayPanel)
    bpy.utils.unregister_class(ExportCamerasAndImagesOperator)
    bpy.utils.unregister_class(ExportSceneAgentPosesOperator)
    bpy.utils.unregister_class(GeneratePreMadeObjectOperator)
    bpy.utils.unregister_class(AddGeometryNodesPointCloudOperator)
    bpy.utils.unregister_class(Render4DGSAnimationOperator)
    del bpy.types.Scene.my_tool

    if handler_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(handler_update)
        print("Camera Array Tool: Depsgraph handler unregistered.")

    if on_new_file in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_new_file)
        print("Camera Array Tool: Load handler unregistered.")


if __name__ == "__main__":
    register()
