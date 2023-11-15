// worker.js
import {
  show_image,
  show_image2,
  get_components,
  get_count,
  overlap_matrix,
  sort_overlap_matrix,
  show_image_depth,
} from "./flat_map.mjs";

self.addEventListener("message", async function (e) {
  if (e.data.type === "image") {
    function update_progress(progress) {
      self.postMessage({ type: "progress", progress });
    }
    update_progress(0);
    let data32_index;
    if (e.data.data_select === "none")
      data32_index = await show_image(e.data, update_progress);
    else data32_index = await show_image_depth(e.data, update_progress);

    let matrix_overlap = null;
    let sort_index = null;
    if (e.data.show_matrix) {
      matrix_overlap = await overlap_matrix(e.data);
      [matrix_overlap, sort_index] = await sort_overlap_matrix(
        matrix_overlap,
        e.data.component_ids,
        e.data.matrix_select,
      );
    }
    //const data32 = await voxels_to_flatmap(data32_index);

    // Post the result back to the main thread
    self.postMessage({
      type: "image",
      data32_index,
      matrix_overlap: matrix_overlap,
      component_ids: e.data.component_ids,
      matrix_select: e.data.matrix_select,
      sort_index,
      update_id: e.data.update_id,
    });
  }
  if (e.data.type === "image2") {
    const data32_index = await show_image2(e.data);

    // Post the result back to the main thread
    self.postMessage({ type: "image", data32_index });
  }
  if (e.data.type === "pixel") {
    let pixel = await get_components(e.data);

    let counts = {};
    for (let index in e.data.component_ids) {
      let count = await get_count({
        component_id: e.data.component_ids_array[index],
        ...e.data,
      });
      counts[e.data.component_ids[index]] = count;
    }

    // Post the result back to the main thread
    self.postMessage({ pixel: pixel, type: "pixel", counts: counts });
  }
});
