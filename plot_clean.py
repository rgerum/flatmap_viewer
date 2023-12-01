import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from pathlib import Path
import time
import json


subject_ids = [0, 1, 2, 3, 4, 5, 6, 7]
#space = 'fsavg'
space = 'fssubject'
roi_names = {'prf-visualrois': ['V1', 'V2', 'V3', 'V4'],
 'floc-bodies': ['mTL-bodies', 'FBA-1', 'FBA-2', 'EBA'],
 'floc-faces': ['aTL-faces', 'mTL-faces', 'FFA-2', 'FFA-1', 'OFA'],
 'floc-places': ['RSC', 'PPA', 'OPA'],
 'floc-words': ['mTL-words', 'mfs-words', 'VWFA-2VWFA-1', 'OWFA']}

roi_list = ['V1', 'V2', 'V3', 'V4', 'OPA', 'RSC', 'PPA', 'EBA', 'FFA'] + roi_names['floc-words']

all_component_ids = [
         0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
        13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
        26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
        39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
        52,  55,  56,  57,  58,  59,  60,  63,  64,  65,  66,  67,  68,
        70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,
        83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
        96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108,
       109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
       122, 123, 124, 125, 126, 127]


run_name = "run-23"
if not os.path.exists("cache"):
    os.mkdir("cache")


def data_to_flatmap(data, subject_id):
    import cortex
    from cortex.quickflat.view import composite

    # create the vertex data object
    subject_name = f'subj0{subject_id + 1}'
    braindata = cortex.dataset.Vertex(data, subject_name if space == 'fssubject' else 'fsaverage', cmap='jet',
                                      vmin=0, vmax=8)
    # and create the flatmap image
    im, extents = composite.make_flatmap_image(braindata, height=1024)
    return im


def load_subject_component(subject_id, component_id):
    # get name and path
    subject_name = f'subj0{subject_id + 1}'
    subject_path = Path(run_name) / subject_name

    # load mask
    lh_data = np.load(subject_path / f'mask__component-{component_id}__{space}__lh.npy')
    rh_data = np.load(subject_path / f'mask__component-{component_id}__{space}__rh.npy')

    # concatenate both hemispheres
    data = np.concatenate([lh_data, rh_data], axis=1)

    # prepare data
    data[data == -1] = np.nan
    data[data > 0] = 1
    data[data < 0] = 1
    #print("data", data.shape, data.dtype, np.unique(data))
    #exit()
    data2 = np.nanmax(data, axis=0)
    data[np.isnan(data)] = 0

    return data, data2

def save_gzip(data, filename):
    import gzip
    with gzip.GzipFile(filename, "w") as f:
        np.save(f, data)

def get_bit_lookout(subject_ids, min_subject_overlap_count):
    def count_bits(number, positions):
        count = 0
        for position in positions:
            if (number & (1 << position)) != 0:
                count += 1
        return count

    bit_count_table = [count_bits(i, subject_ids) >= min_subject_overlap_count for i in range(256)]
    return bit_count_table

def get_voxel_count(a, layer_count):
    voxel_count = 327684
    layer_ids_offsets = [
        0,
        327684,
        655368,
        983052,
        1310736
    ][:layer_count]
    mask_new = np.zeros(voxel_count, dtype=np.uint8)
    bitCountTable = get_bit_lookout(subject_ids, 4)
    for i in range(voxel_count):
        mask_pix = 0
        for index_layer_offset in layer_ids_offsets:
            mask_pix |= a[i + index_layer_offset]
        mask_new[i] = bitCountTable[mask_pix]
    return mask_new


def load_all_new(output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for index, component_id in enumerate(all_component_ids):
        print(component_id)
        data_all = []
        # iterate over subjects
        for subject_id in subject_ids:
            # load the data
            dataAllLayers, data = load_subject_component(subject_id, component_id)
            data_all.append(dataAllLayers.ravel().astype(np.uint8) << subject_id)

        mask = np.sum(data_all, axis=0).astype(np.uint8)

        np.save(Path(output_folder) / f"mask_data_{index}.npy", mask)


def cache_mapping_voxel_pixel_old(output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    component_id = all_component_ids[0]
    subject_id = subject_ids[0]

    # load the data
    dataAllLayers, data = load_subject_component(subject_id, component_id)

    x = np.arange(data.shape[0])
    data2 = data_to_flatmap(x, subject_id)
    data2[np.isnan(data2)] = -1
    plt.imshow(data2)
    plt.show()
    np.save(Path(output_folder) / "mapping_map.npy", data2.astype(np.int32).ravel())


def cache_mapping_voxel_pixel(output_folder, subject_id):
    import cortex
    from cortex.quickflat.view import composite

    x = np.arange(100)

    # create the vertex data object
    space = 'fssubject'
    if subject_id == -1:
        subject_name = 'fsaverage'
        subject_id = 1
    subject_name = f'subj0{subject_id}'

    try:
        braindata = cortex.dataset.Vertex(x, subject_name if space == 'fssubject' else 'fsaverage', cmap='jet',
                                          vmin=0, vmax=8)
    except ValueError as e:
        import re
        m = re.match(r".*or (\d*) for both.*", str(e))
        x = np.arange(int(m.group(1)))
        braindata = cortex.dataset.Vertex(x, subject_name if space == 'fssubject' else 'fsaverage', cmap='jet',
                                          vmin=0, vmax=8)
    # and create the flatmap image
    im, extents = composite.make_flatmap_image(braindata, height=1024)

    data2 = im

    data2[np.isnan(data2)] = -1
    #fig,axs = plt.subplots(1, 2, sharex=True, sharey=True)
    #plt.sca(axs[0])
    #plt.imshow(data2)
    #plt.sca(axs[1])
    #cache_mapping_voxel_pixel_old(output_folder)
    #plt.show()
    np.save(Path(output_folder) / "mapping_map.npy", data2.astype(np.int32))


def load_all_new_mask(output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    component_id = all_component_ids[0]
    data_masks = []
    # iterate over subjects
    for subject_id in subject_ids:
        # load the data
        dataAllLayers, data = load_subject_component(subject_id, component_id)
        data_masks.append(~np.isnan(data))

    mask_all = np.zeros(data_masks[0].shape, dtype=np.int64)
    for index, mask in enumerate(data_masks):
        mask_all += mask << index
    np.save(Path(output_folder) / f"data_masks_all.npy", mask_all.astype(np.uint8))


def cache_flatmap_background(output_folder):
    height = 1024
    subject_id = 0
    subject_name = f'subj0{subject_id + 1}'

    _, initial = load_subject_component(0, 0)
    import cortex
    import nibabel as nib
    from cortex.quickflat.view import composite

    braindata = cortex.dataset.Vertex(initial, subject_name if space == 'fssubject' else 'fsaverage',
                                      cmap='jet',
                                      vmin=0, vmax=8, )
    overlay_file = f'{subject_name}/overlays_version1.svg' if space == 'fssubject' else f'fsaverage/overlays_floc.svg'

    print(cortex.options.usercfg)
    print(cortex.database.default_filestore)
    print(Path(cortex.database.default_filestore) / overlay_file)
    cortex.quickflat.make_figure(braindata, with_rois=True, with_curvature=True, with_colorbar=True,
                                 colorbar_location=(0.01, 0.05, 0.2, 0.05),
                                 height=height,
                                 overlay_file=Path(cortex.database.default_filestore) / overlay_file,
                                 roi_list=roi_list)
    images = [child for child in plt.gcf().axes[0].get_children() if isinstance(child, mpl.image.AxesImage)]
    im1 = np.array(images[1].get_array())
    im1 = np.array([im1, im1, im1, im1 * 0 + 1]).transpose(1, 2, 0).copy(order='C')
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    plt.imsave(Path(output_folder) / "background.png", im1)
    plt.imsave(Path(output_folder) / "foreground.png", np.array(images[2].get_array()))


def cache_images(output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    #im = plt.imread("run-9/top_images_73k.png")
    im = plt.imread("top_images_test.png")

    if 0:
        count = 113
        im2 = im[count * 256:(count + 1) *256]
        plt.imshow(im2)
        plt.show()
        exit()

    w, h, c = im.shape
    if not os.path.exists("cache_top_image"):
        os.mkdir("cache_top_image")
    for i in range(0, w, 256):
        for j in range(0, h, 256):
            plt.imsave(Path(output_folder) / f"{i // 256}_{j // 256}.png", im[i:i + 256, j:j + 256, :][::2, ::2])

def cache_component_list(output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    with open(Path(output_folder) / "all_component_ids.json", "w") as f:
        json.dump({"all_component_ids": all_component_ids, "subject_ids": subject_ids}, f)


def rotate_point_around_axis(point, axis, theta):
    """
    Rotates a point in 3D space around a given axis ('x', 'y', or 'z').

    :param point: List or array-like representing the point [x, y, z]
    :param axis: Char representing the axis ('x', 'y', or 'z')
    :param theta: Angle of rotation in radians
    :return: Rotated point as a list [x', y', z']
    """
    # Ensure the point is a NumPy array
    point = np.asarray(point)

    # Define the rotation matrices for each axis
    if axis.lower() == 'x':
        R = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])
    elif axis.lower() == 'y':
        R = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])
    elif axis.lower() == 'z':
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("The axis must be 'x', 'y', or 'z'")

    # Multiply the rotation matrix with the point to get the rotated point
    rotated_point = np.dot(point, R)

    return rotated_point


def save_3D_data(surf_name="fsaverage", output_folder="static_data"):
    import cortex

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    curv_vertices = cortex.db.get_surfinfo(surf_name)
    np.save(output_folder / f"curvature.npy", curv_vertices.data)

    index_mapping = None

    counts = {}

    # flat, inflated, pia, wm
    def store_3d_data(name, f=1, name_vertex=None):
        pt, vtx = cortex.db.get_surf(surf_name, name, merge=True, nudge=True)
        if name == "flat":
            pt1, vtx1 = cortex.db.get_surf(surf_name, name, hemisphere="lh", merge=True, nudge=True)
            pt2, vtx2 = cortex.db.get_surf(surf_name, name, hemisphere="rh", merge=True, nudge=True)
            print(pt1.shape, pt2.shape, vtx1.shape, vtx2.shape)
            counts["lh_count"] = pt1.shape[0]
            counts["rh_count"] = pt2.shape[0]
        pt = pt / 100 * f
        print(pt.shape, pt.dtype, vtx.dtype, np.mean(pt, axis=0), np.max(pt, axis=0), np.min(pt, axis=0))
        # center
        pt -= (np.max(pt, axis=0) + np.min(pt, axis=0)) / 2

        # rotate as the viewer has z axis pointing out of the plane
        if name != "flat":
            # pt = np.ascontiguousarray(pt[:, [2, 1, 0]])
            pt = rotate_point_around_axis(pt, "z", np.pi / 2)
            pt = rotate_point_around_axis(pt, "x", np.pi / 2)
            pt = rotate_point_around_axis(pt, "y", np.pi / 2)
            #pt[:, 0] *= -1
            pt = np.ascontiguousarray(pt, dtype=np.float32)

            # Define a key function to use with np.argsort that checks if a row from list2 is in set1
            def sort_key(row):
                return (tuple(row) not in set1,)

            # Create an index array that will sort list2 with common rows first
            contained = np.apply_along_axis(sort_key, 1, vtx)[:, 0]
            #print(contained.shape, np.mean(contained))
            indices = np.argsort(contained)

            # Apply the sorting to list2
            #print(indices.shape, indices)
            vtx = np.asarray(vtx[indices])
            #print(vtx)
            #exit()
        else:
            pt[:, 0] *= -1

        print(pt.shape, pt.dtype, vtx.shape, vtx.dtype)
        if index_mapping is not None and 0:
            # Apply the mapping to the faces to update the indices
            vtx = index_mapping[vtx.ravel()].reshape(vtx.shape).astype(np.int32)

            # Reorder the pt array according to the new order
            pt = pt[new_order]
        print(pt.shape, pt.dtype, vtx.shape, vtx.dtype)
        print("----------")
        name1 = output_folder / f"pt_{name}.npy"
        np.save(name1, pt)
        if name_vertex is not None:
            name2 = output_folder / f"{name_vertex}.npy"
            np.save(name2, vtx)
            return name1, name2

    name1, name2 = store_3d_data("flat", name_vertex="faces_flat")

    pt = np.load(name1)
    faces = np.load(name2)
    # Identify the unique indices used by the faces
    used_indices = np.unique(faces.ravel())

    # Generate the complete set of point indices
    all_indices = np.arange(len(pt))

    print(len(used_indices), len(all_indices))

    # Identify the unused indices
    unused_indices = np.setdiff1d(all_indices, used_indices)

    # Create a new order for the points, where used points come first
    new_order = np.concatenate((used_indices, unused_indices))

    # Create a mapping from old indices to new indices
    index_mapping = np.argsort(new_order)

    set1 = set(map(tuple, faces))

    #store_3d_data("flat", name_vertex="vtx_flat")
    store_3d_data("inflated", name_vertex="faces")
    store_3d_data("pia", f=1.2)
    store_3d_data("wm", f=1.25)
    with open(output_folder / "counts.json", "w") as f:
        json.dump(counts, f)


def print_colormap(cmap_name):
    cmap = plt.get_cmap(cmap_name)
    data = cmap(np.linspace(0, 1, 256))
    print(list(data[:,:3].ravel()))


if __name__ == "__main__":
    save_3D_data("fsaverage", f"static_data/fsaverage")
    for i in range(1, 9):
        save_3D_data(f"subj0{i}", f"static_data/subj0{i}")
    cache_mapping_voxel_pixel(f"static_data/fsaverage", -1)
    for i in range(1, 9):
        cache_mapping_voxel_pixel(f"static_data/subj0{i}", i)


    exit()
    height = 1024
    width = 2274
    def get_connections(i):
        data = []
        for di in [-1, width, -width, 1]:
            try:
                data.append(mask[i+di])
            except IndexError:
                pass
        return np.unique(data)

    output_folder = "static_data/component_masks"
    mask = np.load(Path(output_folder) / "mapping_map.npy")
    data_masks_all = np.load(Path(output_folder) / "data_masks_all.npy")
    connections = {}
    print(mask.shape)
    print(np.max(mask))
    for i in np.unique(mask):
        if i == -1:
            continue
        if data_masks_all[i] == 0:
            continue
        print(i)
        points = np.where(mask == i)[0]
        if len(points) == 0:
            exit()
            connections.append([])
            continue
        #print(points, len(points))
        all = np.unique(np.hstack([get_connections(p) for p in points]))
        all = [x for x in all if i != x]
        connections[i] = all
        continue
        print(all)

        m = (mask == i).astype(np.int64)
        for j in range(len(all)):
            print(all[j], np.sum(mask == all[j]))
            m[mask == all[j]] += j+3
            print(np.unique(m))
        plt.imshow(m.reshape(height, width))
        plt.show()
    print(connections)
    import pickle
    with open("static_data/connections.pkl", "wb") as f:
        pickle.dump(connections, f)
    exit()
        #connections.append()

    print(mask.shape, np.max(mask), np.min(mask))
    exit()

    #cache_component_list("static_data")
    #cache_flatmap_background("static_data")
    #cache_images("static_data/component_example_images")

    #load_all_new(f"static_data/component_masks/{run_name}")
    #cache_mapping_voxel_pixel("static_data/component_masks")
    #load_all_new_mask("static_data/component_masks")

    #save_3D_data()
    print_colormap("viridis")


    exit()


    im = plt.imread("static_data/background.png")
    pt = np.load("static_data/pt_flat.npy")
    vtx = np.load("static_data/vtx_flat.npy")
    print(vtx.shape)
    print(vtx.ravel()[:6])
    #exit()
    plt.imshow(im, extent=[0, 1, 1, 0])
    w, h = 512, 512
    im2 = np.linspace(0, 1, im[:, :, 0].ravel().shape[0]).reshape(im[:, :, 0].T.shape).T
    im2b = np.linspace(0, 1, im[:, :, 0].ravel().shape[0]).reshape(im[:, :, 0].shape)

    im2 = np.linspace(0, 1, w*h).reshape((w, h)).T
    im2b = np.linspace(0, 1, w*h).reshape((h, w))
    imm = np.dstack([im2, im2b, im2*0])
    print(imm.shape)
    plt.imsave("static_data/background2.png", imm)
    f = 358
    f1 = -f/im.shape[1]
    f2 = -f/im.shape[0]
    print(f1, f2)
    idx = vtx.ravel()[:3]
    plt.plot(pt[:, 0]*f1+0.5, pt[:, 1]*f2+0.5, 'r.')
    plt.show()