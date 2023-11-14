import { cachedLoadNpy } from "./numpy_to_js.mjs";

let [height, width] = [1024, 2274]; //data_masks_all.shape;
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

function convertIndexToBits(subject_ids) {
  let all_bits = 0;
  for (let id of subject_ids) {
    all_bits |= 1 << id;
  }
  return all_bits;
}

async function loadAllNpyInParallel(component_ids_array, runs) {
  let promises = [];
  for (let comp of component_ids_array)
    promises.push(
      cachedLoadNpy(
        `../static_data/component_masks/${runs}/mask_data_${comp}.npy`,
      ),
    );

  return await Promise.all(promises);
}

function getBitCountTable(subject_ids, min_subject_overlap_count) {
  const bitCountTable = new Uint8Array(256);
  for (let i = 0; i < 256; i++) {
    bitCountTable[i] = countBits(i, subject_ids) >= min_subject_overlap_count;
  }
  return bitCountTable;
}

export async function get_components({
  component_ids_array,
  component_ids,
  subject_ids,
  min_subject_overlap_count,
  layer_ids,
  voxel,
  runs,
}) {
  let layer_ids_offsets = layer_ids.map((x) => x * voxel_count);

  let data_arrays = await loadAllNpyInParallel(component_ids_array, runs);
  let components = [];
  let i = voxel;

  for (let j in data_arrays) {
    let a = data_arrays[j].data;
    let mask_pix = 0;
    for (let index_layer_offset of layer_ids_offsets) {
      mask_pix |= a[i + index_layer_offset];
    }
    if (countBits(mask_pix, subject_ids) >= min_subject_overlap_count) {
      components.push(component_ids[parseInt(j)]);
    }
  }
  return components;
}

export async function get_count({
  component_id,
  subject_ids,
  min_subject_overlap_count,
  layer_ids,
  runs,
}) {
  const bitCountTable = getBitCountTable(
    subject_ids,
    min_subject_overlap_count,
  );

  let layer_ids_offsets = layer_ids.map((x) => x * voxel_count);

  let data_array = await loadAllNpyInParallel([component_id], runs);
  let a = data_array[0].data;
  let count = 0;
  for (let i = 0; i < voxel_count; i++) {
    let mask_pix = 0;
    for (let index_layer_offset of layer_ids_offsets) {
      mask_pix |= a[i + index_layer_offset];
    }
    count += bitCountTable[mask_pix];
  }
  return count;
}

export async function overlap_matrix({
  component_ids_array,
  subject_ids,
  min_subject_overlap_count,
  layer_ids,
  runs,
}) {
  const all_bits = convertIndexToBits(subject_ids);
  const bitCountTable = getBitCountTable(
    subject_ids,
    min_subject_overlap_count,
  );

  console.time("LoadBinary");
  const data_arrays = await loadAllNpyInParallel(component_ids_array, runs);
  const data_masks_all = await cachedLoadNpy(
    "../static_data/component_masks/data_masks_all.npy",
  );
  console.timeEnd("LoadBinary");

  const voxel_count = data_masks_all.shape[0];

  const data_masks_all_d = data_masks_all.data;

  const layer_ids_offsets = layer_ids.map((x) => x * voxel_count);

  const component_count = data_arrays.length;
  const matrix_overlap = new Int32Array(component_count * component_count);

  const current_maps = [];

  console.time("group");
  for (let j = 0; j < component_count; j++) {
    const map = new Uint8Array(voxel_count);
    const a = data_arrays[j].data;
    for (let i = 0; i < voxel_count; i++) {
      if (!(data_masks_all_d[i] & all_bits)) {
        continue;
      }

      let mask_pix = 0;
      for (let index_layer_offset of layer_ids_offsets) {
        mask_pix |= a[i + index_layer_offset];
      }
      map[i] = bitCountTable[mask_pix];
    }
    current_maps.push(map);
  }
  console.timeEnd("group");

  console.time("group2");
  for (let i = 0; i < voxel_count; i++) {
    for (let x = 0; x < component_count; x++) {
      if (current_maps[x][i]) {
        matrix_overlap[x * component_count + x] += 1;
        for (let y = x + 1; y < component_count; y++) {
          if (current_maps[y][i]) {
            matrix_overlap[x * component_count + y] += 1;
            matrix_overlap[y * component_count + x] += 1;
          }
        }
      }
    }
  }
  console.timeEnd("group2");
  return matrix_overlap;
}

function argsortDesc(arr) {
  // Create an array of indices [0, 1, ..., arr.length - 1].
  const indices = Array.from(arr.keys());

  // Sort the indices array based on comparing values in the arr.
  indices.sort((a, b) => arr[b] - arr[a]);

  return indices;
}

export async function sort_overlap_matrix(
  matrix_overlap,
  component_ids,
  matrix_select,
) {
  let component_count = Math.sqrt(matrix_overlap.length);
  if (!matrix_select || matrix_select == "none") {
    return [matrix_overlap, component_ids];
  }
  let matrix_overlap_sorted = new Int32Array(component_count * component_count);
  let values_to_sort = [];
  for (let i = 0; i < component_count; i++)
    values_to_sort.push(
      matrix_overlap[
        i * component_count + component_ids.indexOf(matrix_select)
      ],
    );

  let indices = argsortDesc(values_to_sort);

  let component_ids_sorted = [];
  for (let i = 0; i < component_count; i++) {
    for (let j = 0; j < component_count; j++) {
      matrix_overlap_sorted[i * component_count + j] =
        matrix_overlap[indices[i] * component_count + indices[j]];
    }
    component_ids_sorted.push(component_ids[indices[i]]);
  }
  return [matrix_overlap_sorted, component_ids_sorted];
}

export async function show_image({
  component_ids_array,
  subject_ids,
  min_subject_overlap_count,
  layer_ids,
  runs,
  cmap_max,
}) {
  const all_bits = convertIndexToBits(subject_ids);
  const bitCountTable = getBitCountTable(
    subject_ids,
    min_subject_overlap_count,
  );

  console.time("LoadBinary");
  const data_arrays = await loadAllNpyInParallel(component_ids_array, runs);
  const data_masks_all = await cachedLoadNpy(
    "../static_data/component_masks/data_masks_all.npy",
  );
  console.timeEnd("LoadBinary");

  const voxel_count = data_masks_all.shape[0];

  const data32_index = new Int32Array(voxel_count);
  const data_arrays_d = data_arrays.map((x) => x.data);

  const data_masks_all_d = data_masks_all.data;
  const maxColorIndex = cmap_max;

  const layer_ids_offsets = layer_ids.map((x) => x * voxel_count);

  console.time("PixelManipulationX");
  for (let i = 0; i < voxel_count; i++) {
    if (!(data_masks_all_d[i] & all_bits)) {
      data32_index[i] = -1;
      continue;
    }

    let bitsCount = 0;
    for (let a of data_arrays_d) {
      let mask_pix = 0;
      for (let index_layer_offset of layer_ids_offsets) {
        mask_pix |= a[i + index_layer_offset];
      }
      bitsCount += bitCountTable[mask_pix];
      if (bitsCount === maxColorIndex) break;
    }

    data32_index[i] = bitsCount;
  }
  console.timeEnd("PixelManipulationX");

  return data32_index;
}

export async function show_image_depth({
  component_ids_array,
  subject_ids,
  min_subject_overlap_count,
  layer_ids,
  runs,
  data_select,
  min_val,
  max_val,
  mean_val,
  range_val,
  data_select2,
}) {
  console.log("show_image_depth", data_select);
  const all_bits = convertIndexToBits(subject_ids);
  const bitCountTable = getBitCountTable(
    subject_ids,
    min_subject_overlap_count,
  );

  let data_depth;
  if (data_select == "min") data_depth = min_val;
  else if (data_select == "max") data_depth = max_val;
  else if (data_select == "mean") data_depth = mean_val;
  else if (data_select == "range") data_depth = range_val;

  console.time("LoadBinary");
  const data_arrays = await loadAllNpyInParallel(component_ids_array, runs);
  const data_masks_all = await cachedLoadNpy(
    "../static_data/component_masks/data_masks_all.npy",
  );
  console.timeEnd("LoadBinary");

  const voxel_count = data_masks_all.shape[0];

  const data32_index = new Int32Array(voxel_count);
  const data_arrays_d = data_arrays.map((x) => x.data);

  const data_masks_all_d = data_masks_all.data;
  const maxColorIndex = 8;

  const layer_ids_offsets = layer_ids.map((x) => x * voxel_count);

  let bitsValue = 0;
  let bitsCount = 0;
  let init_agg = function () {
    bitsValue = 0;
    bitsCount = 0;
  };
  let add_agg = function (value) {
    bitsCount += 1;
    bitsValue += value;
  };

  let sum_agg = function () {
    return bitsValue / bitsCount;
  };

  if (data_select2 == "min") {
    init_agg = function () {
      bitsValue = Infinity;
    };
    add_agg = function (value) {
      if (value < bitsValue) bitsValue = value;
    };
    sum_agg = function () {
      return bitsValue;
    };
  } else if (data_select2 == "max") {
    init_agg = function () {
      bitsValue = -Infinity;
    };
    add_agg = function (value) {
      if (value > bitsValue) bitsValue = value;
    };
    sum_agg = function () {
      return bitsValue;
    };
  }

  console.time("PixelManipulationX");
  for (let i = 0; i < voxel_count; i++) {
    if (!(data_masks_all_d[i] & all_bits)) {
      data32_index[i] = -1;
      continue;
    }

    init_agg();
    for (let ii in data_arrays_d) {
      let a = data_arrays_d[ii];
      let mask_pix = 0;
      for (let index_layer_offset of layer_ids_offsets) {
        mask_pix |= a[i + index_layer_offset];
      }
      if (bitCountTable[mask_pix]) add_agg(data_depth[component_ids_array[ii]]);
      //bitsCount += bitCountTable[mask_pix];
      //bitsValue += data_depth[component_ids_array[ii]] * bitCountTable[mask_pix];
      //if (bitsCount === maxColorIndex)
      //    break;
    }

    data32_index[i] = sum_agg();
  }
  console.timeEnd("PixelManipulationX");

  return data32_index;
}

export async function show_image2({
  component_index2,
  subject_ids,
  min_subject_overlap_count,
  layer_ids,
  runs,
}) {
  let all_bits = convertIndexToBits(subject_ids);
  const bitCountTable = getBitCountTable(
    subject_ids,
    min_subject_overlap_count,
  );

  console.time("LoadBinary");
  let list_data_arrays = [];
  for (let comp of component_index2) {
    let data_array = await loadAllNpyInParallel(comp, runs);
    list_data_arrays.push(data_array);
  }
  const data_masks_all = await cachedLoadNpy(
    "../static_data/component_masks/data_masks_all.npy",
  );
  console.timeEnd("LoadBinary");

  let voxel_count = data_masks_all.shape[0];

  const data32_index = new Int32Array(voxel_count);

  let list_data_arrays_d = list_data_arrays.map((x) => x.map((y) => y.data));
  let data_masks_all_d = data_masks_all.data;
  const maxColorIndex = colors.length - 1;

  let layer_ids_offsets = layer_ids.map((x) => x * voxel_count);

  console.time("PixelManipulationX");
  for (let i = 0; i < voxel_count; i++) {
    if (!(data_masks_all_d[i] & all_bits)) {
      data32_index[i] = -1;
      continue;
    }

    let bitsCount2 = 0;
    for (let data_arrays_d of list_data_arrays_d) {
      let bitsCount = 0;
      for (let a of data_arrays_d) {
        let mask_pix = 0;
        for (let index_layer_offset of layer_ids_offsets) {
          mask_pix |= a[i + index_layer_offset];
        }
        bitsCount += bitCountTable[mask_pix];
        if (bitsCount) break;
      }
      bitsCount2 += bitsCount;
      if (bitsCount2 === maxColorIndex) break;
    }

    data32_index[i] = bitsCount2;
  }
  console.timeEnd("PixelManipulationX");

  return data32_index;
}
