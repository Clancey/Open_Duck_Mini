"""3D viewport controls for the Open Duck Mini animation workflow."""

import bpy
from bpy.props import StringProperty


def register_properties():
    bpy.types.Scene.duck_export_directory = StringProperty(
        name="Export directory", subtype="DIR_PATH", default="//"
    )


def unregister_properties():
    del bpy.types.Scene.duck_export_directory


class DUCK_PT_animation_panel(bpy.types.Panel):
    bl_label = "Open Duck Mini"
    bl_idname = "DUCK_PT_animation_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Duck"

    def draw(self, context):
        layout = self.layout
        layout.operator("duck.build_rig", icon="ARMATURE_DATA")
        layout.separator()
        layout.prop(context.scene, "duck_export_directory")
        rig = context.active_object
        action = rig.animation_data.action if rig and rig.animation_data else None
        if action:
            for key, default, label in (
                ("duck_loop", False, "Loop"), ("duck_blend_in", 0.25, "Blend In"),
                ("duck_blend_out", 0.25, "Blend Out"), ("duck_priority", 10, "Priority"),
                ("duck_layer", "override", "Layer"), ("duck_tags", "", "Tags"),
            ):
                if key not in action:
                    action[key] = default
                layout.prop(action, f'["{key}"]', text=label)
        operator = layout.operator("duck.export_clip", icon="EXPORT")
        operator.output_directory = context.scene.duck_export_directory
