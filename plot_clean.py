import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from pathlib import Path
import time
import json


subject_ids = [0, 1, 2, 3, 4, 5, 6, 7]
space = 'fsavg'
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

#component_ids = np.array([  0,   1,   2,   3,   4])
#component_ids = np.array([  0, 1, 2])

run_name = "run-22"
if not os.path.exists("cache"):
    os.mkdir("cache")


def flatten_nested_list(nested_list):
    # Initialize an empty list to store the flattened elements
    flattened_list = []

    # Iterate through the outer list
    for sublist in nested_list:
        # Iterate through each inner list
        for item in sublist:
            # Append each item to the flattened list
            flattened_list.append(item)

    return flattened_list

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


def load_all():
    data_all_components = []
    data_masks = []
    mapping = None

    for component_id in all_component_ids:
        print(component_id)
        # if we have not yet cached the data
        if 1:#not os.path.exists(f'cache/component-{component_id}.npy'):
            # we create a flatmap that contains the set bit for which subjects have a mask there for this component
            data_all = []
            # iterate over subjects
            for subject_id in subject_ids:
                # load the data
                dataAllLayers, data = load_subject_component(subject_id, component_id)

                #print(dataAllLayers.shape, dataAllLayers[0, :100], np.unique(dataAllLayers), data.shape, data[ :100])
                #dataAllLayers[0] = data
                #exit()
                # create a flatmap from the data

                if mapping is None:
                    print(data.shape)
                    x = np.arange(data.shape[0])
                    #np.random.shuffle(x)
                    print(x.shape, x.dtype)
                    data2 = data_to_flatmap(x, subject_id)
                    #print(x.shape[0]/2)
                    #exit()
                    data2 = data2.astype(int).flatten()

                    def index_to_coordinates_mapping(arr):
                        # Find the maximum index in the array to determine the size of the mapping list
                        max_index = np.max(arr)

                        # Initialize a list of empty lists to store the coordinates
                        mapping = [[0] for _ in range(max_index + 1)]

                        # Iterate through the numpy array and append the coordinates to the corresponding index
                        for i in range(arr.shape[0]):
                            index = arr[i]
                            if index > 0:
                                mapping[index].append(i)

                        return mapping

                    mapping = index_to_coordinates_mapping(data2)


                #data = data_to_flatmap(data, subject_id)

                # store the valid voxel mask but only once for each subject
                if component_id == 0:
                    data_masks.append(~np.isnan(data))
                # store the data, encoded as the nth bit, n being the subject number
                data_all.append(dataAllLayers.ravel().astype(np.uint8) << subject_id)
            # sum the data to create the bit mask
            data_all = np.sum(data_all, axis=0).astype(np.uint8)
            # store it
            np.save(f'cache/component-{component_id}.npy', data_all)
            # store the valid voxel mask data (only for component 0 as they are the same across components)
            if component_id == 0:
                np.save(f'cache/component-{component_id}_data_masks.npy', data_masks)
        else:
            # if we already have calculated the data, just load it
            data_all = np.load(f'cache/component-{component_id}.npy')
            if component_id == 0:
                data_masks = np.load(f'cache/component-{component_id}_data_masks.npy')
        data_all_components.append(data_all)

    data_all_components = np.stack(data_all_components)
    print(data_all_components.shape)
    return data_all_components, data_masks, mapping

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


def cache_masks(output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    mask_data, data_masks, mapping = load_all()

    mapping = np.array(flatten_nested_list(mapping), np.uint32)
    np.save(Path(output_folder) / f"mapping.npy", mapping)
    #with open(Path(output_folder) / "mapping.json", "w") as f:
    #    json.dump(mapping, f)

    for index, mask in enumerate(mask_data):
        np.save(Path(output_folder) /f"mask_data_{index}.npy", mask)
        import gzip
        with gzip.GzipFile(Path(output_folder) /f"mask_data_{index}.npy.gz", "w") as f:
            np.save(f, mask)

    mask_all = np.zeros(data_masks[0].shape, dtype=np.int64)
    for index, mask in enumerate(data_masks):
        mask_all += mask << index
    np.save(Path(output_folder) / f"data_masks_all.npy", mask_all.astype(np.uint8))
    import gzip
    with gzip.GzipFile(Path(output_folder) / f"data_masks_all.npy.gz", "w") as f:
        np.save(f, mask_all.astype(np.uint8))
    print("turbo", [list((np.array(plt.get_cmap("turbo")(i / 8)) * 255).astype(np.uint8)) for i in range(9)])

def create_static_content():
    #cache_images()
    pass

class Plotter:
    def __init__(self):
        self.mask_data, self.data_masks = load_all()

        if 0:
            for index, mask in enumerate(self.mask_data):
                import gzip
                with gzip.GzipFile(f"cache_masks/mask_data_{index}.npy.gz", "w") as f:
                    np.save(f, mask)

            mask_all = np.zeros(self.data_masks[0].shape, dtype=np.int64)
            for index, mask in enumerate(self.data_masks):
                mask_all += mask << index
            import gzip
            with gzip.GzipFile(f"cache_masks/data_masks_all.npy.gz", "w") as f:
                np.save(f, mask_all.astype(np.uint8))
            print("turbo", [list((np.array(plt.get_cmap("turbo")(i/8))*255).astype(np.uint8)) for i in range(9)])

        height = 1024
        subject_id = 0
        subject_name = f'subj0{subject_id + 1}'

        if os.path.exists("cache/layer_1.npy"):
            im1 = np.load("cache/layer_1.npy")
            im2 = np.load("cache/layer_2.npy")
            self.im1 = np.array([im1, im1, im1, im1*0+1]).transpose(1, 2, 0).copy(order='C')
            self.im2 = im2

            if 0:
                overlay_file = f'{subject_name}/overlays_version1.svg' if space == 'fssubject' else f'fsaverage/overlays_floc.svg'
                import cortex
                print("overlay_file", Path(cortex.database.default_filestore) / overlay_file)
                import shutil
                shutil.copy(Path(cortex.database.default_filestore) / overlay_file, "cache_masks/overlay.svg")
        else:
            initial = load_subject_component(0, 0)
            import cortex
            import nibabel as nib
            from cortex.quickflat.view import composite
            braindata = cortex.dataset.Vertex(initial, subject_name if space == 'fssubject' else 'fsaverage',
                                              cmap='jet',
                                              vmin=0, vmax=8, )
            overlay_file = f'{subject_name}/overlays_version1.svg' if space == 'fssubject' else f'fsaverage/overlays_floc.svg'

            print(cortex.options.usercfg)
            print(cortex.database.default_filestore)
            cortex.quickflat.make_figure(braindata, with_rois=True, with_curvature=True, with_colorbar=True,
                                         colorbar_location=(0.01, 0.05, 0.2, 0.05),
                                         height=height,
                                         overlay_file=Path(cortex.database.default_filestore) / overlay_file,
                                         roi_list=roi_list)
            images = [child for child in plt.gcf().axes[0].get_children() if isinstance(child, mpl.image.AxesImage)]
            for i, im in enumerate(images):
                np.save(f"cache/layer_{i}.npy", np.array(im.get_array()))
                print("im", im.get_array().shape, im.get_clim(), im.get_extent())
            plt.show()

    def plot(self, component_ids, subject_ids, min_subject_overlap_count):
        t = time.time()

        im1 = np.zeros_like(self.mask_data[0])
        masks = np.zeros_like(self.mask_data[0], dtype=bool)
        # iterate over all the components
        for component_id in component_ids:
            # get the id of the component in the list
            component_index = all_component_ids.index(component_id)

            im = np.zeros_like(self.mask_data[0])
            masks = np.zeros_like(self.mask_data[0], dtype=bool)
            # iterate over all subject
            for subject_id in subject_ids:
                # select which subject is present in the bit mask
                im += (self.mask_data[component_index] & (1 << subject_id)).astype(bool)
                masks = masks | self.data_masks[subject_id]
            # filter by number of subjects and add to total mask
            im1 += im >= min_subject_overlap_count

        im1 = im1.astype(np.float32)
        im1[~masks] = np.nan
        print("load", time.time() - t)

        im = plt.get_cmap("turbo")(im1 / 8)
        return im

    def get_components(self, x, y):
        x = int(self.mask_data[0].shape[1] * x / 100)
        y = int(self.mask_data[0].shape[0] * y / 100)

        components = {}
        components_all = []
        for index, mask in enumerate(self.mask_data):
            id = all_component_ids[index]
            value = mask[y, x]
            if value > 0:
                components[id] = [subject_id for subject_id in subject_ids if value & (1 << subject_id)]
            components_all.append([int((value & (1 << subject_id)) != 0) for subject_id in subject_ids])
        return x, y, components, components_all


def cache_images(output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    im = plt.imread("run-9/top_images_73k.png")
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


if __name__ == "__main__":
    #cache_component_list("static_data")
    #cache_flatmap_background("static_data")
    #cache_images("static_data/component_example_images")
    #cache_masks("static_data/component_masks")

    load_all_new("static_data/component_masks")
    exit()
    import cortex

    curv_vertices = cortex.db.get_surfinfo("fsaverage")
    np.save(f"static_data/curvature.npy", curv_vertices.data)

    # flat, inflated, pia, wm
    def store_3d_data(name, f=1):
        pt, vtx = cortex.db.get_surf("fsaverage", name, merge=True, nudge=True)
        print(pt.shape, pt.dtype, np.mean(pt, axis=0), np.max(pt, axis=0), np.min(pt, axis=0))
        # center
        pt -= (np.max(pt, axis=0)+np.min(pt, axis=0))/2

        # rotate as the viewer has z axis pointing out of the plane
        #if name != "flat":
        #pt = np.ascontiguousarray(pt[:, [2, 1, 0]])
        pt = rotate_point_around_axis(pt, "z", np.pi/2)
        pt = rotate_point_around_axis(pt, "x", np.pi/2)
        pt = rotate_point_around_axis(pt, "y", np.pi/2)
        pt[:, 0] *= -1
        pt = np.ascontiguousarray(pt, dtype=np.float32)

        np.save(f"static_data/pt_{name}.npy", pt/100*f)
        #np.save(f"static_data/vtx.npy", vtx)


    #store_3d_data("flat")
    #store_3d_data("pia", f=1.2)
    store_3d_data("wm", f=1.25)
    #[store_3d_data(name) for name in ["flat", "inflated", "pia", "wm"]]
