const colors = [[48, 18, 59, 255], [70, 107, 227, 255], [40, 187, 235, 255], [50, 241, 151, 255], [164, 252, 59, 255], [237, 207, 57, 255], [250, 125, 32, 255], [208, 47, 4, 255], [122, 4, 2, 255]];
let packedColor = [];
for(let i = 0; i < colors.length; i++) {
    let color = colors[i];
    packedColor.push((color[3] << 24) | (color[2] << 16) | (color[1] << 8) | color[0]);
}

// Function to count the number of bits set in the number for the given positions
function countBits(number, positions) {
    let count = 0;
    for (let position of positions) {
        if ((number & (1 << position)) !== 0) {
            count++;
        }
    }
    return count;
}

const cached = {};
async function cachedLoadNpy(url) {
    if(url in cached) {
        return cached[url];
    }
    const data = await loadNpy(url);
    cached[url] = data;
    return data;
}


function convertIndexToBits(subject_ids) {
    let all_bits = 0;
    for(let id of subject_ids) {
        all_bits |= 1 << id;
    }
    return all_bits
}

async function loadAllNpyInParallel(component_ids_array) {
    let promises = [];
    for(let comp of component_ids_array)
        promises.push(cachedLoadNpy("../static_data/component_masks/mask_data_" + comp + ".npy"));

    return await Promise.all(promises);
}

async function get_components(component_ids_array, component_ids, subject_ids, min_subject_overlap_count, x, y) {
    let all_bits = convertIndexToBits(subject_ids);
    let data_arrays = await loadAllNpyInParallel(component_ids_array);
    const data_masks_all = await cachedLoadNpy("../static_data/component_masks/data_masks_all.npy");
    let components = [];
    let i = y * data_masks_all.shape[1] + x;

    for(let j in data_arrays) {
        let array = data_arrays[j];
        if(countBits(array.data[i], subject_ids) >= min_subject_overlap_count) {
            components.push(component_ids[parseInt(j)]);
        }
    }
    console.log("components", components, x, y)
    return components
}

async function get_count(component_id, subject_ids, min_subject_overlap_count) {
    console.log("get_count", component_id, subject_ids, min_subject_overlap_count)
    const bitCountTable = new Uint8Array(256);
    for (let i = 0; i < 256; i++) {
        bitCountTable[i] = countBits(i, subject_ids) >= min_subject_overlap_count;
    }

    let data_array = await loadAllNpyInParallel([component_id]);
    let a = data_array[0].data;
    let width = data_array[0].shape[1];
    let height = data_array[0].shape[0];
    let count = 0;
    for (let i = 0; i < width * height; i++) {
        count += bitCountTable[a[i]];
    }
    return count
}

async function show_image(component_ids_array, subject_ids, min_subject_overlap_count) {
    console.time("LoadBinary");

    let all_bits = convertIndexToBits(subject_ids);
    let data_arrays = await loadAllNpyInParallel(component_ids_array);

    const data_masks_all = await cachedLoadNpy("../static_data/component_masks/data_masks_all.npy");

    console.timeEnd("LoadBinary");

    let width = data_masks_all.shape[1];
    let height = data_masks_all.shape[0];

    let data32 = new Uint32Array(width * height);

    const bitCountTable = new Uint8Array(256);
    for (let i = 0; i < 256; i++) {
        bitCountTable[i] = countBits(i, subject_ids) >= min_subject_overlap_count;
    }
    let data_arrays_d = []
    for(let array of data_arrays) {
        data_arrays_d.push(array.data);
    }

    let data_masks_all_d = data_masks_all.data
    const maxColorIndex = colors.length - 1;

    console.time("PixelManipulationX");
    for (let i = 0; i < width * height; i++) {
        if (!(data_masks_all_d[i] & all_bits))
            continue

        let bitsCount = 0;
        for(let a of data_arrays_d) {
            bitsCount += bitCountTable[a[i]];
            if(bitsCount == maxColorIndex)
                break;
        }

        data32[i] = packedColor[bitsCount];
    }
    console.timeEnd("PixelManipulationX");

    return data32;
}