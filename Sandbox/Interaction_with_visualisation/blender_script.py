import bpy
import mathutils
import math
import threading
import time

parameters_file = r'D:\Code\NEvol_git\Sandbox\Interaction_with_visualisation\parameters.txt'  # Путь к файлу параметров

# Глобальные переменные для хранения состояния
external_params = {
    'is_moving': False,
    'is_rotating_left': False,
    'is_rotating_right': False
}

def read_external_parameters():
    global external_params
    while True:
        with open(parameters_file, 'r') as f:
            params = f.read().strip().split(',')
            if len(params) == 3:
                external_params['is_moving'] = params[0] == 'True'
                external_params['is_rotating_left'] = params[1] == 'True'
                external_params['is_rotating_right'] = params[2] == 'True'
        time.sleep(0.5)  # Чтение параметров каждые 0.5 секунды

def move_object():
    obj = bpy.context.object
    if obj is None:
        return 0.1  # Продолжать

    forward_speed = 0.04
    rotation_angle = math.radians(15)  # Угол поворота - 15 градусов

    # Если объект движется вперед
    if external_params['is_moving']:
        direction = obj.matrix_world.to_quaternion() @ mathutils.Vector((0.0, -forward_speed, 0.0))
        obj.location += direction

    # Если объект поворачивается влево
    if external_params['is_rotating_left']:
        obj.rotation_euler.z += rotation_angle
        external_params['is_rotating_left'] = False
        update_parameters_file()

    # Если объект поворачивается вправо
    if external_params['is_rotating_right']:
        obj.rotation_euler.z -= rotation_angle
        external_params['is_rotating_right'] = False
        update_parameters_file()

    return 0.1  # Повторять каждые 0.1 секунды

def update_parameters_file():
    with open(parameters_file, 'w') as f:
        f.write(f"{external_params['is_moving']},{external_params['is_rotating_left']},{external_params['is_rotating_right']}")

def register():
    bpy.utils.register_class(OBJECT_OT_move_and_rotate)
    bpy.ops.object.move_and_rotate('INVOKE_DEFAULT')
    bpy.app.timers.register(move_object, first_interval=0.1)

def unregister():
    bpy.utils.unregister_class(OBJECT_OT_move_and_rotate)
    if move_object in bpy.app.timers:
        bpy.app.timers.unregister(move_object)

class OBJECT_OT_move_and_rotate(bpy.types.Operator):
    """Move and Rotate Object"""
    bl_idname = "object.move_and_rotate"
    bl_label = "Move and Rotate Object"

    def execute(self, context):
        return {'RUNNING_MODAL'}

if __name__ == "__main__":
    threading.Thread(target=read_external_parameters, daemon=True).start()
    register()