import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from pathlib import Path
import time


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

run_name = "run-9"
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
    data = np.nanmax(data, axis=0)

    return data

def load_all():
    data_all_components = []
    data_masks = []

    for component_id in all_component_ids:
        # if we have not yet cached the data
        if not os.path.exists(f'cache/component-{component_id}.npy'):
            # we create a flatmap that contains the set bit for which subjects have a mask there for this component
            data_all = []
            # iterate over subjects
            for subject_id in subject_ids:
                # load the data
                data = load_subject_component(subject_id, component_id)
                # create a flatmap from the data
                data = data_to_flatmap(data, subject_id)
                # store the valid voxel mask but only once for each subject
                if component_id == 0:
                    data_masks.append(~np.isnan(data))
                # store the data, encoded as the nth bit, n being the subject number
                data_all.append(data.astype(np.uint8) << subject_id)
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
    return data_all_components,  data_masks


class Plotter:
    def __init__(self):
        self.mask_data, self.data_masks = load_all()

        height = 1024
        subject_id = 0
        subject_name = f'subj0{subject_id + 1}'

        if os.path.exists("cache/layer_1.npy"):
            im1 = np.load("cache/layer_1.npy")
            im2 = np.load("cache/layer_2.npy")
            self.im1 = np.array([im1, im1, im1, im1*0+1]).transpose(1, 2, 0).copy(order='C')
            self.im2 = im2
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



if __name__ == "__main__":
    #load_all()
    Plotter()
    #plot(3, subject_ids)