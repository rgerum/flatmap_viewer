import { loadNpy } from "./numpy_to_js.mjs";
import { get_cmap_uint32 } from "./colormaps.mjs";
import { cachedLoadNpy } from "./numpy_to_js.mjs";
import { overlap_matrix } from "./flat_map.mjs";
import { drawHeatmap, drawMatrix2 } from "./draw_matrix.mjs";

let mapping = undefined;
let mapping_inverse = undefined;

let [height, width] = [1024, 2274]; //data_masks_all.shape;
let voxel_count = 327684;

async function getMapping() {
  console.time("LoadBinary3");
  if (!mapping) {
    mapping_inverse = (
      await loadNpy("static_data/component_masks/mapping_map.npy")
    ).data;
    mapping = [];
    for (let i = 0; i < voxel_count; i++) {
      mapping.push([]);
    }
    for (let i = 0; i < width * height; i++) {
      let index = mapping_inverse[i];
      if (index >= 0) mapping[mapping_inverse[i]].push(i);
    }
  }
  console.timeEnd("LoadBinary3");
  return [mapping, mapping_inverse];
}

async function getPixelVoxel(x, y) {
  let i0 = y * width + x;
  let [mapping, mapping_inverse] = await getMapping();
  let voxel = mapping_inverse[i0];
  return voxel;
}

async function getVoxelPixel({ voxel }) {
  try {
    let index = mapping[voxel][0];
    let x = index % width;
    let y = Math.floor(index / width);
    return [x, y];
  } catch (e) {
    return [-1, -1];
  }
}

export async function voxels_to_flatmap(data32_index, cmap_max) {
  console.time("voxels_to_flatmap");
  let [mapping, mapping_inverse] = await getMapping();

  const curvature = (await cachedLoadNpy("static_data/curvature.npy")).data;

  let data32 = new Uint32Array(width * height);

  let packedColor = get_cmap_uint32("turbo", cmap_max);
  let packedColor2 = get_cmap_uint32("gray", 4);
  const maxColorIndex = packedColor.length - 1;

  for (let i = 0; i < data32_index.length; i++) {
    let clr;
    if (data32_index[i] >= 0)
      clr = packedColor[Math.min(data32_index[i], maxColorIndex)];
    else clr = curvature[i] > 0 ? packedColor2[2] : packedColor2[1];

    for (let ii of mapping[i]) {
      data32[ii] = clr;
    }
  }
  console.timeEnd("voxels_to_flatmap");

  let result_img = new Uint8ClampedArray(data32.buffer);
  result_img.shape = [height, width, 4];
  return result_img;
}

export function add_2D_view(dom_elem) {
  let dom_centering = document.createElement("div");
  dom_centering.classList.add("image_centering");
  dom_elem.appendChild(dom_centering);

  let img_background = document.createElement("img");
  img_background.src = "static_data/background.png";
  img_background.classList.add("image");
  dom_centering.appendChild(img_background);

  let canvas = document.createElement("canvas");
  canvas.id = "myCanvas";
  canvas.width = 2274;
  canvas.height = 1024;
  canvas.classList.add("image");
  dom_centering.appendChild(canvas);

  let img_foreground = document.createElement("img");
  img_foreground.src = "static_data/foreground.png";
  img_foreground.classList.add("image");
  dom_centering.appendChild(img_foreground);

  let canvasPoint = document.createElement("canvas");
  canvasPoint.id = "myCanvasPoint";
  canvasPoint.width = 2274;
  canvasPoint.height = 1024;
  canvasPoint.classList.add("image");
  dom_centering.appendChild(canvasPoint);

  // Initialize panzoom
  const plotImage = dom_elem;
  const panzoom = Panzoom(plotImage, { canvas: true, maxScale: 10 });
  const parent = plotImage.parentElement;
  // No function bind needed
  parent.addEventListener("wheel", panzoom.zoomWithWheel);

  // This demo binds to shift + wheel
  parent.addEventListener("wheel", function (event) {
    if (!event.shiftKey) return;
    panzoom.zoomWithWheel(event);
  });

  let selected_pos = [-1, -1];
  let zoom_scale = 1;

  function updatePointDisplay() {
    const [x, y] = selected_pos;
    let canvas = canvasPoint;
    let ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const line_width = Math.max(4 / zoom_scale, 1);
    if (x !== -1 || y !== -1) {
      ctx.beginPath();
      ctx.arc(x, y, line_width, 0, 2 * Math.PI, false);
      ctx.rect(0, y, canvas.width, line_width);
      ctx.rect(x, 0, line_width, canvas.height);
      ctx.fillStyle = "red";
      ctx.fill();
    }
  }

  plotImage.addEventListener("panzoomzoom", (event) => {
    zoom_scale = event.detail.scale;
    updatePointDisplay();
  });

  canvasPoint.addEventListener("click", async function (event) {
    const rect = canvasPoint.getBoundingClientRect();

    // Calculate click position as percentage of bounding box dimensions
    const xPercent = ((event.clientX - rect.left) / rect.width) * 100;
    const yPercent = ((event.clientY - rect.top) / rect.height) * 100;
    let x = parseInt((xPercent * 2274) / 100);
    let y = parseInt((yPercent * 1024) / 100);

    let voxel = await getPixelVoxel(x, y);
    var myEvent = new CustomEvent("voxel_selected_changed", {
      detail: { voxel: voxel, x: x, y: y },
    });
    window.dispatchEvent(myEvent);
  });

  async function set_voxel_selected({ voxel, x, y }) {
    if (x === undefined) [x, y] = await getVoxelPixel({ voxel });
    if (voxel === -1) {
      x = -1;
      y = -1;
    }
    selected_pos = [x, y];
    document.getElementsByName("x")[0].value = x;
    document.getElementsByName("y")[0].value = y;
    updatePointDisplay();
  }

  async function set_voxel_data(image, cmap_max) {
    const data32 = await voxels_to_flatmap(image, cmap_max);
    const processedImageData = new ImageData(
      data32,
      canvas.width,
      canvas.height,
    );
    let ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.putImageData(processedImageData, 0, 0);

    set_texture(data32, canvas.width, canvas.height);
  }

  let last_overlay_matrix = null;
  let last_sort_index = null;
  let click_event_to_x_y = null;
  async function plot_overlap_matrix({
    matrix_overlap,
    component_ids,
    matrix_select,
    sort_index,
  }) {
    last_overlay_matrix = matrix_overlap;
    last_sort_index = sort_index;
    /*
        let canvas = document.getElementById("matrix");
        let w = Math.sqrt(matrix_overlap.length);
        let ctx = canvas.getContext("2d");
        canvas.width = w;
        canvas.height = w;
        let data = new Uint8ClampedArray(matrix_overlap.length * 4);

        let max = Math.max(...matrix_overlap);
        for (let i = 0; i < matrix_overlap.length; i++) {
            let c= matrix_overlap[i] /max * 255
            data[i * 4] = c;
            data[i * 4 + 1] = c;
            data[i * 4 + 2] = c;
            data[i * 4 + 3] = 255;
        }

        const processedImageData = new ImageData(data, w, w);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.putImageData(processedImageData, 0, 0);

         */

    //click_event_to_x_y = drawHeatmap(canvas, matrix_overlap, sort_index, max);
    drawMatrix2(matrix_overlap, sort_index, (x, y, i, j, v) => {
      const form = document.getElementById("plotForm");
      const formData = new URLSearchParams(new FormData(form));
      const component_ids = formData.getAll("component_ids");

      document.getElementById("matrix_clicked").innerText =
        i + " " + j + " " + v;

      //var myEvent = new CustomEvent('voxel_selected_changed', {detail: {voxel: -1}});
      //window.dispatchEvent(myEvent);

      var myEvent2 = new CustomEvent("display_components", {
        detail: { components: [i, j] },
      });
      window.dispatchEvent(myEvent2);
    });

    // Array of options to add
    var options = ["none"].concat(component_ids);

    function checkOptions(selectId, comparisonList) {
      // Get the select element
      const selectElement = document.getElementById(selectId);

      // Retrieve values from the select element's options
      let selectValues = [];
      for (let i = 0; i < selectElement.options.length; i++) {
        selectValues.push(selectElement.options[i].value); // or .text if you need the text
      }

      // Check if each option is different from the given list
      let differences = selectValues.filter(
        (value) => !comparisonList.includes(value),
      );

      return differences;
    }

    // Get the select element
    var select = document.getElementById("matrix_select");
    select.innerHTML = "";

    // Add options to the select element
    options.forEach(function (option) {
      var opt = document.createElement("option");
      opt.value = option;
      opt.innerHTML = option;
      select.appendChild(opt);
    });
    select.selectedIndex = component_ids.indexOf(matrix_select) + 1;
  }

  /*
    let canvas_matrix = document.getElementById("matrix");
    canvas_matrix.addEventListener("click", async function (event) {
        if(!click_event_to_x_y)
            return
        let [x, y] = click_event_to_x_y(event);
        if(x === -1 || y === -1)
            return

        const form = document.getElementById("plotForm");
        const formData = new URLSearchParams(new FormData(form));
        const component_ids = formData.getAll("component_ids");

        document.getElementById("matrix_clicked").innerText = last_sort_index[x] + " " + last_sort_index[y] + " " + last_overlay_matrix[x * Math.sqrt(last_overlay_matrix.length) + y];

        //var myEvent = new CustomEvent('voxel_selected_changed', {detail: {voxel: -1}});
        //window.dispatchEvent(myEvent);

        var myEvent2 = new CustomEvent('display_components', {detail: {components: [last_sort_index[x], last_sort_index[y]]}});
        window.dispatchEvent(myEvent2);
    });*/

  return {
    set_voxel_data,
    set_voxel_selected,
    plot_overlap_matrix,
  };
}
