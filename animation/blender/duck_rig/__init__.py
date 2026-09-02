"""Open Duck Mini Blender authoring tools."""

bl_info = {
    "name": "Open Duck Mini Rig and Clip Exporter",
    "author": "Open Duck Mini contributors",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "View3D > Sidebar > Duck",
    "description": "Build an URDF-based rig and export .duckanim.json clips",
    "category": "Animation",
}

from .rig_builder import DUCK_OT_build_rig
from .exporter import DUCK_OT_export_clip
from .panel import DUCK_PT_animation_panel

CLASSES = (DUCK_OT_build_rig, DUCK_OT_export_clip, DUCK_PT_animation_panel)


def register():
    for cls in CLASSES:
        __import__("bpy").utils.register_class(cls)
    from .panel import register_properties

    register_properties()


def unregister():
    from .panel import unregister_properties

    unregister_properties()
    for cls in reversed(CLASSES):
        __import__("bpy").utils.unregister_class(cls)

