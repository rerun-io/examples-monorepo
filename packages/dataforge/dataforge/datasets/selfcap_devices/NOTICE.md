# SelfCap device meshes

Display meshes for the head-worn devices, logged by the SelfCap ingest. Each is in its device's reference
camera frame (RDF: x right, y down, z out of the lens; metres). Arrays: `positions` float32, `triangles`
uint32, `normals` float16, `colors` uint8 RGB.

- `oak_d_pro_w.npz`: the OAK-D Pro W enclosure, RGB lens at the origin. From Luxonis' `OAK-D-PRO-W.stl`
  (depthai-ros, MIT License, Copyright (c) Luxonis Holding Corporation), welded and decimated from
  246,306 to 26,696 triangles.
- `quest3s.npz`: a Meta Quest 3S stand-in in its left RGB camera frame, built from measured dimensions
  and the d31 factory baseline (63.55 mm). Not a Meta model.
