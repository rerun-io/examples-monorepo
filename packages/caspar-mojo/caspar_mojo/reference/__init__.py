"""Float64 numpy oracle of CASPAR's bundle adjustment: camera models, pose math, the LM + PCG solver, test problems.

Every function also runs in float32 (it follows the dtype of its inputs), so a float32 run tells float32 effects
apart from bugs in the Mojo solver.
"""
