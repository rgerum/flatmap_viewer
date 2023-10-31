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

async function get_components(component_ids_array, component_ids, subject_ids, min_subject_overlap_count, layer_ids, x, y) {
    let [mapping, mapping_inverse] = await getMapping();
    let layer_ids_offsets = layer_ids.map((x, i) => i * voxel_count + x);

    let all_bits = convertIndexToBits(subject_ids);
    let data_arrays = await loadAllNpyInParallel(component_ids_array);
    const data_masks_all = await cachedLoadNpy("../static_data/component_masks/data_masks_all.npy");
    let components = [];
    let i0 = y * width + x
    console.log("i0", i0, y, width, x);
    let i = mapping_inverse[i0];
    console.log("i", i);

    for(let j in data_arrays) {
        let array = data_arrays[j];
        for(let index_layer_offset of layer_ids_offsets) {
            if(countBits(array.data[i+index_layer_offset], subject_ids) >= min_subject_overlap_count) {
                components.push(component_ids[parseInt(j)]);
                break
            }
        }

        //if(countBits(array.data[i], subject_ids) >= min_subject_overlap_count) {
        //    components.push(component_ids[parseInt(j)]);
        //}
    }
    return components
}

async function get_count(component_id, subject_ids, min_subject_overlap_count, layer_ids) {
    const bitCountTable = new Uint8Array(256);
    for (let i = 0; i < 256; i++) {
        bitCountTable[i] = countBits(i, subject_ids) >= min_subject_overlap_count;
    }

    let layer_ids_offsets = layer_ids.map((x, i) => i * voxel_count + x);

    let data_array = await loadAllNpyInParallel([component_id]);
    let a = data_array[0].data;
    let count = 0;
    for (let i = 0; i < voxel_count; i++) {
        for(let index_layer_offset of layer_ids_offsets) {
            if(bitCountTable[a[i+index_layer_offset]]) {
                count += 1;
                break
            }
        }
        //count += bitCountTable[a[i]];
    }
    return count
}

let mapping = undefined;
let mapping_inverse = undefined;
async function getMapping() {
    if(!mapping) {
        mapping = await (await fetch("../static_data/component_masks/mapping.json")).json();

        let [height, width] = [1024, 2274]//data_masks_all.shape;
        let voxel_count = 327684;
        mapping_inverse = new Uint32Array(width * height);
        for(let i = 0; i < voxel_count; i++) {
            for(let mm of mapping[i]) {
                mapping_inverse[mm] = i;
            }
        }
    }
    return [mapping, mapping_inverse]
}


async function show_image(component_ids_array, subject_ids, min_subject_overlap_count, layer_ids) {
    console.time("LoadBinary2");
    let [mapping, mapping_inverse] = await getMapping();
    console.timeEnd("LoadBinary2");

    console.time("LoadBinary");

    let all_bits = convertIndexToBits(subject_ids);
    let data_arrays = await loadAllNpyInParallel(component_ids_array);

    const data_masks_all = await cachedLoadNpy("../static_data/component_masks/data_masks_all.npy");

    console.timeEnd("LoadBinary");

    let [height, width] = [1024, 2274]//data_masks_all.shape;
    let voxel_count = data_masks_all.shape[0];

    let data32 = new Uint32Array(width * height);

    const bitCountTable = new Uint8Array(256);
    for (let i = 0; i < 256; i++) {
        bitCountTable[i] = countBits(i, subject_ids) >= min_subject_overlap_count;
    }
    let data_arrays_d = data_arrays.map(x => x.data);

    let data_masks_all_d = data_masks_all.data
    const maxColorIndex = colors.length - 1;

    let layer_ids_offsets = layer_ids.map((x, i) => i * voxel_count + x);

    console.time("PixelManipulationX");
    for(let i = 0; i < voxel_count; i++) {
        if (!(data_masks_all_d[i] & all_bits))
                continue

        let bitsCount = 0;
        for(let a of data_arrays_d) {
            for(let index_layer_offset of layer_ids_offsets) {
                if(bitCountTable[a[i+index_layer_offset]]) {
                    bitsCount += 1;
                    break
                }
            }
            if(bitsCount === maxColorIndex)
                break;
        }

        for (let ii of mapping[i]) {
            data32[ii] = packedColor[bitsCount];
        }
    }
    console.timeEnd("PixelManipulationX");

    return data32;
}

async function show_image2(list_component_ids_array, subject_ids, min_subject_overlap_count, layer_ids) {
    console.time("LoadBinary2");
    let mapping = await getMapping();
    console.timeEnd("LoadBinary2");

    console.time("LoadBinary");

    let all_bits = convertIndexToBits(subject_ids);

    let list_data_arrays = []
    for(let comp of list_component_ids_array) {
        let data_array = await loadAllNpyInParallel(comp);
        list_data_arrays.push(data_array);
    }
    const data_masks_all = await cachedLoadNpy("../static_data/component_masks/data_masks_all.npy");

    console.timeEnd("LoadBinary");

    let [height, width] = [1024, 2274]//data_masks_all.shape;
    let voxel_count = data_masks_all.shape[0];

    let data32 = new Uint32Array(width * height);

    const bitCountTable = new Uint8Array(256);
    for (let i = 0; i < 256; i++) {
        bitCountTable[i] = countBits(i, subject_ids) >= min_subject_overlap_count;
    }
    let list_data_arrays_d = list_data_arrays.map(x => x.map(y => y.data));

    let data_masks_all_d = data_masks_all.data
    const maxColorIndex = colors.length - 1;

     let layer_ids_offsets = layer_ids.map((x, i) => i * voxel_count + x);

    console.time("PixelManipulationX");
    for (let i = 0; i < width * height; i++) {
        if (!(data_masks_all_d[i] & all_bits))
            continue

        let bitsCount2 = 0;
        for(let data_arrays_d of list_data_arrays_d) {
            let bitsCount = 0;
            for(let a of data_arrays_d) {
                bitsCount += bitCountTable[a[i]];
                if(bitsCount)
                    break;
            }
            bitsCount2 += bitsCount;
            if(bitsCount2 == maxColorIndex)
                break;
        }


        data32[i] = packedColor[bitsCount2];
    }
    console.timeEnd("PixelManipulationX");

    return data32;
}