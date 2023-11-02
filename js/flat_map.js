const colors = [[48, 18, 59, 255], [70, 107, 227, 255], [40, 187, 235, 255], [50, 241, 151, 255], [164, 252, 59, 255], [237, 207, 57, 255], [250, 125, 32, 255], [208, 47, 4, 255], [122, 4, 2, 255]];
let packedColor = [];
for(let i = 0; i < colors.length; i++) {
    let color = colors[i];
    packedColor.push((color[3] << 24) | (color[2] << 16) | (color[1] << 8) | color[0]);
}

let [height, width] = [1024, 2274]//data_masks_all.shape;
let voxel_count = 327684;

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

function getBitCountTable(subject_ids, min_subject_overlap_count) {
    const bitCountTable = new Uint8Array(256);
    for (let i = 0; i < 256; i++) {
        bitCountTable[i] = countBits(i, subject_ids) >= min_subject_overlap_count;
    }
    return bitCountTable
}

async function getPixelVoxel(x, y) {
    let i0 = y * width + x;
    let [mapping, mapping_inverse] = await getMapping();
    let voxel = mapping_inverse[i0];
    return voxel
}

async function getVoxelPixel({voxel}) {
    let index = mapping[voxel][0];
    let x = index % width;
    let y = Math.floor(index / width);
    return [x, y]
}

async function get_components({
                                  component_ids_array,
                                  component_ids,
                                  subject_ids,
                                  min_subject_overlap_count,
                                  layer_ids,
                                  x,
                                  y
                              }) {
    let [mapping, mapping_inverse] = await getMapping();
    let layer_ids_offsets = layer_ids.map((x, i) => i * voxel_count + x);

    let data_arrays = await loadAllNpyInParallel(component_ids_array);
    let components = [];
    let i0 = y * width + x;
    let i = mapping_inverse[i0];

    for(let j in data_arrays) {
        let a = data_arrays[j].data;
        let mask_pix = 0;
        for(let index_layer_offset of layer_ids_offsets) {
            mask_pix |= a[i + index_layer_offset]
        }
        if(countBits(mask_pix, subject_ids) >= min_subject_overlap_count) {
            components.push(component_ids[parseInt(j)]);
        }
    }
    return components
}

async function get_count({component_id, subject_ids, min_subject_overlap_count, layer_ids}) {
    const bitCountTable = getBitCountTable(subject_ids, min_subject_overlap_count);

    let layer_ids_offsets = layer_ids.map((x, i) => i * voxel_count);

    let data_array = await loadAllNpyInParallel([component_id]);
    let a = data_array[0].data;
    let count = 0;
    for (let i = 0; i < voxel_count; i++) {
        let mask_pix = 0;
        for(let index_layer_offset of layer_ids_offsets) {
            mask_pix |= a[i + index_layer_offset]
        }
        count += bitCountTable[mask_pix];
    }
    return count
}

let mapping = undefined;
let mapping_inverse = undefined;

async function getMapping() {
    console.time("LoadBinary3");
    if(!mapping) {
        mapping_inverse = (await loadNpy("../static_data/component_masks/mapping_map.npy")).data;
        mapping = [];
        for(let i = 0; i < voxel_count; i++) {
            mapping.push([]);
        }
        for(let i = 0; i < width*height; i++) {
            let index = mapping_inverse[i];
            if(index >= 0)
                mapping[mapping_inverse[i]].push(i);
        }
    }
    console.timeEnd("LoadBinary3");
    return [mapping, mapping_inverse]
}


async function show_image({component_ids_array, subject_ids, min_subject_overlap_count, layer_ids}) {
    let [mapping, mapping_inverse] = await getMapping();

    console.time("LoadBinary");

    let all_bits = convertIndexToBits(subject_ids);
    let data_arrays = await loadAllNpyInParallel(component_ids_array);

    const data_masks_all = await cachedLoadNpy("../static_data/component_masks/data_masks_all.npy");
    const curvature = (await cachedLoadNpy("../static_data/curvature.npy")).data;

    console.timeEnd("LoadBinary");

    let [height, width] = [1024, 2274]//data_masks_all.shape;
    let voxel_count = data_masks_all.shape[0];
    console.log("voxel_count", voxel_count)

    let data32 = new Uint32Array(width * height);
    let data32_colors = new Float64Array(voxel_count * 3);

    const bitCountTable = getBitCountTable(subject_ids, min_subject_overlap_count);
    let data_arrays_d = data_arrays.map(x => x.data);

    let data_masks_all_d = data_masks_all.data
    const maxColorIndex = colors.length - 1;

    let layer_ids_offsets = layer_ids.map((x, i) => i * voxel_count);

    console.time("PixelManipulationX");
    for(let i = 0; i < voxel_count; i++) {
        if (!(data_masks_all_d[i] & all_bits)) {
            if(curvature[i] > 0) {
                data32_colors[i * 3 + 0] = 0.62;
                data32_colors[i * 3 + 1] = 0.62;
                data32_colors[i * 3 + 2] = 0.62;
            }
            else {
                data32_colors[i * 3 + 0] = 0.37;
                data32_colors[i * 3 + 1] = 0.37;
                data32_colors[i * 3 + 2] = 0.37;
            }
            continue
        }

        let bitsCount = 0;
        for(let a of data_arrays_d) {
            let mask_pix = 0
            for(let index_layer_offset of layer_ids_offsets) {
                mask_pix |= a[i + index_layer_offset]
            }
            bitsCount += bitCountTable[mask_pix];
            if(bitsCount === maxColorIndex)
                break;
        }

        for (let ii of mapping[i]) {
            data32[ii] = packedColor[bitsCount];
        }
        data32_colors[i * 3 + 0] = colors[bitsCount][0]/255;
        data32_colors[i * 3 + 1] = colors[bitsCount][1]/255;
        data32_colors[i * 3 + 2] = colors[bitsCount][2]/255;
    }
    console.timeEnd("PixelManipulationX");

    return [data32, data32_colors];
}

async function show_image2({component_index2, subject_ids, min_subject_overlap_count, layer_ids}) {
    let [mapping, mapping_inverse] = await getMapping();

    console.time("LoadBinary");

    let all_bits = convertIndexToBits(subject_ids);

    let list_data_arrays = []
    for(let comp of component_index2) {
        let data_array = await loadAllNpyInParallel(comp);
        list_data_arrays.push(data_array);
    }
    const data_masks_all = await cachedLoadNpy("../static_data/component_masks/data_masks_all.npy");
    const curvature = (await cachedLoadNpy("../static_data/curvature.npy")).data;

    console.timeEnd("LoadBinary");

    let [height, width] = [1024, 2274]//data_masks_all.shape;
    let voxel_count = data_masks_all.shape[0];

    let data32 = new Uint32Array(width * height);
    let data32_colors = new Float64Array(voxel_count * 3);

    const bitCountTable = getBitCountTable(subject_ids, min_subject_overlap_count);
    let list_data_arrays_d = list_data_arrays.map(x => x.map(y => y.data));

    let data_masks_all_d = data_masks_all.data
    const maxColorIndex = colors.length - 1;

    let layer_ids_offsets = layer_ids.map((x, i) => i * voxel_count + x);

    console.time("PixelManipulationX");
    for (let i = 0; i < voxel_count; i++) {
        if (!(data_masks_all_d[i] & all_bits)) {
            if(curvature[i] > 0) {
                data32_colors[i * 3 + 0] = 0.62;
                data32_colors[i * 3 + 1] = 0.62;
                data32_colors[i * 3 + 2] = 0.62;
            }
            else {
                data32_colors[i * 3 + 0] = 0.37;
                data32_colors[i * 3 + 1] = 0.37;
                data32_colors[i * 3 + 2] = 0.37;
            }
            continue
        }

        let bitsCount2 = 0;
        for(let data_arrays_d of list_data_arrays_d) {
            let bitsCount = 0;
            for(let a of data_arrays_d) {
                let mask_pix = 0;
                for(let index_layer_offset of layer_ids_offsets) {
                    mask_pix |= a[i + index_layer_offset]
                }
                bitsCount += bitCountTable[mask_pix];
                if(bitsCount)
                    break;
            }
            bitsCount2 += bitsCount;
            if(bitsCount2 == maxColorIndex)
                break;
        }

        for (let ii of mapping[i]) {
            data32[ii] = packedColor[bitsCount2];
        }
        data32_colors[i * 3 + 0] = colors[bitsCount2][0]/255;
        data32_colors[i * 3 + 1] = colors[bitsCount2][1]/255;
        data32_colors[i * 3 + 2] = colors[bitsCount2][2]/255;
    }
    console.timeEnd("PixelManipulationX");

    return [data32, data32_colors];
}