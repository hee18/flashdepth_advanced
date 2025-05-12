'''
https://viser.studio/main/conventions/
In viser, all camera parameters exposed to the Python API use the COLMAP/OpenCV convention:

Forward: +Z
Up: -Y
Right: +X

Confusingly, this is different from Nerfstudio, which adopts the OpenGL/Blender convention:

Forward: -Z
Up: +Y
Right: +X
* Note: nerfstudio is c2w (for instance, dl3dv transforms.json is in this format)

Conversion between the two is a simple 180 degree rotation around the local X-axis.
'''

 
def nerfstudio_to_opencv(matrix, c2w=True):
    # Post-multiplies the 4x4 matrix M by a 180-degree rotation around the X-axis.
    # This flips -Z to +Z and +Y to -Y in the local camera frame.

    # 4x4 rotation: 180 degrees around X
    Rx_180 = np.array([
        [ 1,  0,  0, 0],
        [ 0, -1,  0, 0],
        [ 0,  0, -1, 0],
        [ 0,  0,  0, 1]
    ], dtype=float)

    matrix = matrix @ Rx_180 # this is the order if c2w; if w2c then Rx_180 @ matrix

    return matrix