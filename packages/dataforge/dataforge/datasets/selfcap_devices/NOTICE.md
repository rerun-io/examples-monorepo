# SelfCap device meshes

Display meshes for the SelfCap devices, attached as catalog assets by `dataforge-register`. Each is an `.rrd` holding one static `Mesh3D` at `/device`,
in its device's reference camera frame (RDF: x right, y down, z out of the lens; metres).

- `oak_d_pro_w.rrd`: the OAK-D Pro W enclosure, RGB lens at the origin. From Luxonis' `OAK-D-PRO-W.stl`
  (depthai-ros, MIT License, Copyright (c) Luxonis Holding Corporation), welded and decimated from
  246,306 to 26,696 triangles.
- `quest3s.rrd`: a Meta Quest 3S stand-in in its left RGB camera frame, built from measured dimensions
  and the d31 factory baseline (63.55 mm). Not a Meta model.
- `iphone12.rrd`: a procedural iPhone 12 in landscape, in the phone camera frame with the Ultra Wide lens glass at
  the origin (the phones' assumed recording lens). Built from Apple's published iPhone 12 dimensional drawing
  (2020-10-13) and technical specifications: 146.7 x 71.5 x 7.4 mm body, camera plateau and the two rear lenses.
  Simplified corners and lens rings. Not an Apple model.
