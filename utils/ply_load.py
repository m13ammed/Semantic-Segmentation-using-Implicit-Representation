from cProfile import label
import torch
from pytorch3d.io.ply_io import _load_ply_raw, _get_verts, _PlyData, _check_faces_indices, asdict
from iopath.common.file_io import PathManager
from typing import List, Optional, Tuple
import numpy as np

from  pytorch3d.structures import Meshes

def _load_ply(f, *, path_manager: PathManager) -> Tuple[_PlyData,torch.Tensor, torch.Tensor]:
    """
    Adjusted version of pytorch3D to allow returing the label
    Load the data from a .ply file.

    Args:
        f:  A binary or text file-like object (with methods read, readline,
            tell and seek), a pathlib path or a string containing a file name.
            If the ply file is in the binary ply format rather than the text
            ply format, then a text stream is not supported.
            It is easiest to use a binary stream in all cases.
        path_manager: PathManager for loading if f is a str.

    Returns:
        _PlyData object
    """
    header, elements = _load_ply_raw(f, path_manager=path_manager)

    verts_data = _get_verts(header, elements)

    face = elements.get("face", None)
    if face is not None:
        face_head = next(head for head in header.elements if head.name == "face")
        if (
            len(face_head.properties) != 1
            or face_head.properties[0].list_size_type is None
        ):
            raise ValueError("Unexpected form of faces data.")
        # face_head.properties[0].name is usually "vertex_index" or "vertex_indices"
        # but we don't need to enforce this.

    if face is None:
        faces = None
    elif not len(face):
        # pyre is happier when this condition is not joined to the
        # previous one with `or`.
        faces = None
    elif isinstance(face, np.ndarray) and face.ndim == 2:  # Homogeneous elements
        if face.shape[1] < 3:
            raise ValueError("Faces must have at least 3 vertices.")
        face_arrays = [face[:, [0, i + 1, i + 2]] for i in range(face.shape[1] - 2)]
        faces = torch.LongTensor(np.vstack(face_arrays).astype(np.int64))
    else:
        face_list = []
        for (face_item,) in face:
            if face_item.ndim != 1:
                raise ValueError("Bad face data.")
            if face_item.shape[0] < 3:
                raise ValueError("Faces must have at least 3 vertices.")
            for i in range(face_item.shape[0] - 2):
                face_list.append([face_item[0], face_item[i + 1], face_item[i + 2]])
        faces = torch.tensor(face_list, dtype=torch.int64)

    if faces is not None:
        _check_faces_indices(faces, max_index=verts_data.verts.shape[0])
    vert = elements.get("vertex", None)
    label = torch.Tensor(vert[-1].astype('int16'))
    rgb = torch.Tensor(vert[1][:,:3].astype('int16'))
    #torch.Tensor(elements.get("vertex", None)[-1].astype('int16'))
    
    return _PlyData(**asdict(verts_data), faces=faces, header=header), label, rgb



def load_ply(
    f, *, path_manager: Optional[PathManager] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Adjusted version of pytorch3D to allow returing the label
    
    Load the verts and faces from a .ply file.
    Note that the preferred way to load data from such a file
    is to use the IO.load_mesh and IO.load_pointcloud functions,
    which can read more of the data.

    Example .ply file format:

    ply
    format ascii 1.0           { ascii/binary, format version number }
    comment made by Greg Turk  { comments keyword specified, like all lines }
    comment this file is a cube
    element vertex 8           { define "vertex" element, 8 of them in file }
    property float x           { vertex contains float "x" coordinate }
    property float y           { y coordinate is also a vertex property }
    property float z           { z coordinate, too }
    element face 6             { there are 6 "face" elements in the file }
    property list uchar int vertex_index { "vertex_indices" is a list of ints }
    end_header                 { delimits the end of the header }
    0 0 0                      { start of vertex list }
    0 0 1
    0 1 1
    0 1 0
    1 0 0
    1 0 1
    1 1 1
    1 1 0
    4 0 1 2 3                  { start of face list }
    4 7 6 5 4
    4 0 4 5 1
    4 1 5 6 2
    4 2 6 7 3
    4 3 7 4 0

    Args:
        f:  A binary or text file-like object (with methods read, readline,
            tell and seek), a pathlib path or a string containing a file name.
            If the ply file is in the binary ply format rather than the text
            ply format, then a text stream is not supported.
            It is easiest to use a binary stream in all cases.
        path_manager: PathManager for loading if f is a str.

    Returns:
        verts: FloatTensor of shape (V, 3).
        faces: LongTensor of vertex indices, shape (F, 3).
    """

    if path_manager is None:
        path_manager = PathManager()
    data, labels , rgb= _load_ply(f, path_manager=path_manager)
    faces = data.faces
    if faces is None:
        faces = torch.zeros(0, 3, dtype=torch.int64)

    return data.verts, faces, labels, rgb

def load_mesh_labels(f, device) -> Tuple[Meshes, torch.Tensor]:
    
    verts, faces, labels, rgb = load_ply(f)
    
    meshes = Meshes(verts=[verts], faces=[faces]).to(device)
    
    
    return meshes, labels.to(device), rgb.to(device)
