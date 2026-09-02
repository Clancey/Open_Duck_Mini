"""Blender UI: clip-metadata panel, export options, and operators.

Loaded only inside Blender; ``bpy`` is guarded so the module imports cleanly on
CI, but the operator/panel/property classes are defined only when Blender is
present (they subclass ``bpy.types.*``).

The panel exposes the §5 ``.duckanim`` clip metadata the author sets — ``name``,
``layer_mask``, ``blend_in_s`` / ``blend_out_s``, ``show_blend_in_s`` /
``show_blend_out_s``, ``loop_mode``, ``requires_mode``, ``priority`` — plus the
D3 ``contacts_valid`` toggle and contact-detection parameters, and the export
output directory. Recording uses the deterministic loop
(:class:`open_duck_anim_blender.recorder.DataRecorder`), then writes the 59-float
JSON and compiles the ``.duckanim`` via :mod:`open_duck_anim_blender.export`.
"""

from __future__ import annotations

import os

from open_duck_anim.clip import LAYER_MASKS, LOOP_MODES, REQUIRES_MODES

from . import constraints as constraints_mod
from . import export as export_mod
from .metadata import ClipMetadata, head_envelope_warnings
from .recorder import DataRecorder

try:  # guarded so CI (no Blender) can import this module
    import bpy  # type: ignore
except Exception:  # pragma: no cover - exercised only inside Blender
    bpy = None  # type: ignore


if bpy is not None:  # pragma: no cover - all UI code runs only inside Blender

    def _enum(values):
        return [(v, v, v) for v in values]

    class DuckAnimClipProperties(bpy.types.PropertyGroup):
        name: bpy.props.StringProperty(name="Clip Name", default="untitled_clip")
        layer_mask: bpy.props.EnumProperty(
            name="Layer Mask", items=_enum(LAYER_MASKS), default="head"
        )
        loop_mode: bpy.props.EnumProperty(
            name="Loop Mode", items=_enum(LOOP_MODES), default="wrap"
        )
        requires_mode: bpy.props.EnumProperty(
            name="Requires Mode", items=_enum(REQUIRES_MODES), default="any"
        )
        priority: bpy.props.IntProperty(name="Priority", default=10)
        blend_in_s: bpy.props.FloatProperty(name="Blend In (s)", default=0.35, min=0.0)
        blend_out_s: bpy.props.FloatProperty(name="Blend Out (s)", default=0.35, min=0.0)
        show_blend_in_s: bpy.props.FloatProperty(
            name="Show Blend In (s)", default=0.1, min=0.0
        )
        show_blend_out_s: bpy.props.FloatProperty(
            name="Show Blend Out (s)", default=0.1, min=0.0
        )
        contacts_valid: bpy.props.BoolProperty(
            name="Contacts Valid",
            description="Uncheck for non-stepping clips: writes [0,0] contacts and "
            "FootContactValid=false so training zeroes the contact weight (D3)",
            default=True,
        )
        ground_z: bpy.props.FloatProperty(name="Ground Z (m)", default=0.0)
        contact_threshold: bpy.props.FloatProperty(
            name="Contact Threshold (m)", default=0.01, min=0.0
        )
        output_dir: bpy.props.StringProperty(
            name="Output Dir", subtype="DIR_PATH", default="//duck_mini_data_records"
        )

    def _clip_metadata_from_props(props, source_blend: str) -> ClipMetadata:
        return ClipMetadata(
            name=props.name,
            layer_mask=props.layer_mask,
            loop_mode=props.loop_mode,
            requires_mode=props.requires_mode,
            priority=int(props.priority),
            blend_in_s=float(props.blend_in_s),
            blend_out_s=float(props.blend_out_s),
            show_blend_in_s=float(props.show_blend_in_s),
            show_blend_out_s=float(props.show_blend_out_s),
            source_blend=source_blend,
            contacts_valid=bool(props.contacts_valid),
        )

    class DUCKANIM_OT_apply_constraints(bpy.types.Operator):
        bl_idname = "duckanim.apply_constraints"
        bl_label = "Apply jnt_range Limits"
        bl_description = "Add/refresh Limit Rotation constraints mirroring MJCF jnt_range (idempotent)"

        def execute(self, context):
            try:
                bones = constraints_mod.apply_limit_rotation_constraints()
            except Exception as exc:  # surface rig/name issues to the user
                self.report({"ERROR"}, "Constraint apply failed: %s" % exc)
                return {"CANCELLED"}
            self.report({"INFO"}, "Applied jnt_range limits to %d bones" % len(bones))
            return {"FINISHED"}

    class DUCKANIM_OT_record_export(bpy.types.Operator):
        bl_idname = "duckanim.record_export"
        bl_label = "Record + Export .duckanim"
        bl_description = "Deterministically record the timeline, write 59-float JSON, compile .duckanim"

        def execute(self, context):
            scene = context.scene
            props = scene.duckanim_props

            blend_path = bpy.data.filepath
            source_blend = os.path.basename(blend_path) if blend_path else "open-duck-mini.blend"

            try:
                recorder = DataRecorder(
                    fps=int(scene.render.fps),
                    ground_z=float(props.ground_z),
                    contact_threshold=float(props.contact_threshold),
                    contacts_valid=bool(props.contacts_valid),
                )
                episode = recorder.record()
            except Exception as exc:
                self.report({"ERROR"}, "Recording failed: %s" % exc)
                return {"CANCELLED"}

            # Author-time head safety-envelope warnings (advisory, D13).
            joints = [fr[7:23] for fr in episode["Frames"]]
            warns = head_envelope_warnings(joints)
            for w in warns[:8]:
                self.report({"WARNING"}, w.message())
            if len(warns) > 8:
                self.report({"WARNING"}, "... %d more head-envelope warnings" % (len(warns) - 8))

            meta_obj = _clip_metadata_from_props(props, source_blend)
            try:
                meta = meta_obj.to_compiler_meta()
            except Exception as exc:
                self.report({"ERROR"}, "Invalid clip metadata: %s" % exc)
                return {"CANCELLED"}

            out_dir = bpy.path.abspath(props.output_dir)
            os.makedirs(out_dir, exist_ok=True)
            source_path = os.path.join(out_dir, "%s.source.json" % props.name)
            duckanim_path = os.path.join(out_dir, "%s.duckanim" % props.name)

            try:
                result = export_mod.export_and_compile(
                    episode, meta, source_path, duckanim_path
                )
            except Exception as exc:
                self.report({"ERROR"}, "Compile failed: %s" % exc)
                return {"CANCELLED"}

            self.report(
                {"INFO"},
                "Exported %s (%d frames, sha %s)"
                % (os.path.basename(result["duckanim_path"]),
                   len(episode["Frames"]), result["source_sha256"][:12]),
            )
            return {"FINISHED"}

    class DUCKANIM_PT_panel(bpy.types.Panel):
        bl_category = "Duck Anim"
        bl_label = "Duck Anim Export"
        bl_space_type = "VIEW_3D"
        bl_region_type = "UI"

        def draw(self, context):
            layout = self.layout
            props = context.scene.duckanim_props

            box = layout.box()
            box.label(text="Clip Metadata")
            box.prop(props, "name")
            box.prop(props, "layer_mask")
            box.prop(props, "loop_mode")
            box.prop(props, "requires_mode")
            box.prop(props, "priority")

            box = layout.box()
            box.label(text="Blend Times")
            box.prop(props, "blend_in_s")
            box.prop(props, "blend_out_s")
            box.prop(props, "show_blend_in_s")
            box.prop(props, "show_blend_out_s")

            box = layout.box()
            box.label(text="Foot Contacts (D3)")
            box.prop(props, "contacts_valid")
            row = box.row()
            row.enabled = props.contacts_valid
            row.prop(props, "ground_z")
            row = box.row()
            row.enabled = props.contacts_valid
            row.prop(props, "contact_threshold")

            box = layout.box()
            box.label(text="Rig Limits")
            box.operator("duckanim.apply_constraints")

            box = layout.box()
            box.label(text="Export")
            box.prop(props, "output_dir")
            box.operator("duckanim.record_export")

    _CLASSES = (
        DuckAnimClipProperties,
        DUCKANIM_OT_apply_constraints,
        DUCKANIM_OT_record_export,
        DUCKANIM_PT_panel,
    )

    def register():
        for cls in _CLASSES:
            bpy.utils.register_class(cls)
        bpy.types.Scene.duckanim_props = bpy.props.PointerProperty(
            type=DuckAnimClipProperties
        )

    def unregister():
        del bpy.types.Scene.duckanim_props
        for cls in reversed(_CLASSES):
            bpy.utils.unregister_class(cls)

else:  # pragma: no cover - importable without Blender, but cannot register

    def register():
        raise RuntimeError("open_duck_anim_blender.panels requires Blender (bpy)")

    def unregister():
        raise RuntimeError("open_duck_anim_blender.panels requires Blender (bpy)")
