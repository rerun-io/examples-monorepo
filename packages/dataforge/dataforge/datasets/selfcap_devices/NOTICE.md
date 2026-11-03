# SelfCap device meshes

Display meshes for the SelfCap devices, attached as catalog assets by `dataforge-register`. Each is an `.rrd` holding one static `Mesh3D` (the Quest: an `Asset3D`) at `/device`,
in its device's reference camera frame (RDF: x right, y down, z out of the lens; metres).

- `oak_d_pro_w.rrd`: the OAK-D Pro W enclosure, RGB lens at the origin. From Luxonis' `OAK-D-PRO-W.stl`
  (depthai-ros, MIT License, Copyright (c) Luxonis Holding Corporation), welded and decimated from
  246,306 to 26,696 triangles.
- `quest3s.rrd`: a Meta Quest 3S visual model (shell, cameras, facial interface, soft strap) as a static `Asset3D`
  (GLB with PBR materials and textures) at `/device`, in the left RGB camera frame. From an independent procedural
  recreation built in Blender 4.5 from public reference photographs and published dimensions (quest3s_model_package,
  2026-11-01); not a Meta model, not CAD. Rebuild with `tools/dev/selfcap_quest_model.py <quest_3s.glb>`, which places
  the scene in the camera frame and drops `KHR_texture_transform`, which Rerun refuses.
- `iphone12.rrd`: a procedural iPhone 12 in landscape, in the phone camera frame with the Ultra Wide lens glass at
  the origin (the phones' assumed recording lens). Built from Apple's published iPhone 12 dimensional drawing
  (2020-10-13) and technical specifications: 146.7 x 71.5 x 7.4 mm body, camera plateau and the two rear lenses.
  Simplified corners and lens rings. Not an Apple model.
