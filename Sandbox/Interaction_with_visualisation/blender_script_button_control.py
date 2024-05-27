import bpy
import mathutils
import math

# Auxiliary variables to track the state of movement and rotation
key_state = {
    'is_moving': False,
    'is_rotating_left': False,
    'is_rotating_right': False
}

class OBJECT_OT_move_and_rotate(bpy.types.Operator):
    """Move and Rotate Object"""
    bl_idname = "object.move_and_rotate"
    bl_label = "Move and Rotate Object"

    def modal(self, context, event):
        global key_state

        if event.type in {'ESC'}:
            return {'CANCELLED'}

        if event.type == 'SPACE':
            if event.value == 'PRESS':
                key_state['is_moving'] = not key_state['is_moving']

        if event.type == 'A':
            if event.value == 'PRESS':
                key_state['is_rotating_left'] = True
            elif event.value == 'RELEASE':
                key_state['is_rotating_left'] = False

        if event.type == 'D':
            if event.value == 'PRESS':
                key_state['is_rotating_right'] = True
            elif event.value == 'RELEASE':
                key_state['is_rotating_right'] = False

        obj = context.object

        forward_speed = 0.04  # Reduced movement speed
        turn_radius = 2.0  # Turn radius for arc simulation

        # If the object is moving forward
        if key_state['is_moving']:
            # Direction of movement forward relative to the local Y axis of the object
            direction = obj.matrix_world.to_quaternion() @ mathutils.Vector((0.0, -forward_speed, 0.0))

            if key_state['is_rotating_left']:
                # Turn left in an arc
                angle = forward_speed / turn_radius
                rotation_quat = mathutils.Quaternion((0, 0, 1), angle)
                direction = rotation_quat @ direction

            elif key_state['is_rotating_right']:
                # Turn right in an arc
                angle = -forward_speed / turn_radius
                rotation_quat = mathutils.Quaternion((0, 0, 1), angle)
                direction = rotation_quat @ direction

            obj.location += direction

            # Gradual change of the object's angle
            if key_state['is_rotating_left'] or key_state['is_rotating_right']:
                obj.rotation_euler.z += angle

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

def register():
    bpy.utils.register_class(OBJECT_OT_move_and_rotate)
    bpy.ops.object.move_and_rotate('INVOKE_DEFAULT')

def unregister():
    bpy.utils.unregister_class(OBJECT_OT_move_and_rotate)

if __name__ == "__main__":
    register()
