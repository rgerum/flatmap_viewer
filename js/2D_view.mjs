import {loadNpy} from "./numpy_to_js.mjs";
import {get_cmap_uint32} from "./colormaps.mjs";
import {cachedLoadNpy} from "./numpy_to_js.mjs";
import {overlap_matrix} from "./flat_map.mjs";

let mapping = undefined;
let mapping_inverse = undefined;

let [height, width] = [1024, 2274]//data_masks_all.shape;
let voxel_count = 327684;

async function getMapping() {
    console.time("LoadBinary3");
    if (!mapping) {
        mapping_inverse = (await loadNpy("static_data/component_masks/mapping_map.npy")).data;
        mapping = [];
        for (let i = 0; i < voxel_count; i++) {
            mapping.push([]);
        }
        for (let i = 0; i < width * height; i++) {
            let index = mapping_inverse[i];
            if (index >= 0)
                mapping[mapping_inverse[i]].push(i);
        }
    }
    console.timeEnd("LoadBinary3");
    return [mapping, mapping_inverse]
}

async function getPixelVoxel(x, y) {
    let i0 = y * width + x;
    let [mapping, mapping_inverse] = await getMapping();
    let voxel = mapping_inverse[i0];
    return voxel
}

async function getVoxelPixel({voxel}) {
    try {
        let index = mapping[voxel][0];
        let x = index % width;
        let y = Math.floor(index / width);
        return [x, y]
    } catch (e) {
        return [-1, -1]
    }
}

async function voxels_to_flatmap(data32_index) {
    console.time("voxels_to_flatmap");
    let [mapping, mapping_inverse] = await getMapping();

    const curvature = (await cachedLoadNpy("static_data/curvature.npy")).data;

    let data32 = new Uint32Array(width * height);

    let packedColor = get_cmap_uint32();
    let packedColor2 = get_cmap_uint32("gray", 4);
    const maxColorIndex = packedColor.length - 1;

    for (let i = 0; i < data32_index.length; i++) {
        let clr;
        if (data32_index[i] >= 0)
            clr = packedColor[Math.min(data32_index[i], maxColorIndex)];
        else
            clr = curvature[i] > 0 ? packedColor2[2] : packedColor2[1];

        for (let ii of mapping[i]) {
            data32[ii] = clr;
        }
    }
    console.timeEnd("voxels_to_flatmap");

    return new Uint8ClampedArray(data32.buffer);
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
    const panzoom = Panzoom(plotImage, {canvas: true, maxScale: 10})
    const parent = plotImage.parentElement
    // No function bind needed
    parent.addEventListener('wheel', panzoom.zoomWithWheel)

    // This demo binds to shift + wheel
    parent.addEventListener('wheel', function (event) {
        if (!event.shiftKey) return
        panzoom.zoomWithWheel(event)
    })

    let selected_pos = [-1, -1];
    let zoom_scale = 1;

    function updatePointDisplay() {
        const [x, y] = selected_pos;
        let canvas = canvasPoint;
        let ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        const line_width = Math.max(4 / zoom_scale, 1)
        if (x !== -1 || y !== -1) {
            ctx.beginPath();
            ctx.arc(x, y, line_width, 0, 2 * Math.PI, false);
            ctx.rect(0, y, canvas.width, line_width);
            ctx.rect(x, 0, line_width, canvas.height);
            ctx.fillStyle = 'red';
            ctx.fill();
        }
    }

    plotImage.addEventListener('panzoomzoom', (event) => {
        zoom_scale = event.detail.scale;
        updatePointDisplay();
    })

    canvasPoint.addEventListener("click", async function (event) {
        const rect = canvasPoint.getBoundingClientRect();

        // Calculate click position as percentage of bounding box dimensions
        const xPercent = ((event.clientX - rect.left) / rect.width) * 100;
        const yPercent = ((event.clientY - rect.top) / rect.height) * 100;
        let x = parseInt(xPercent * 2274 / 100);
        let y = parseInt(yPercent * 1024 / 100);

        let voxel = await getPixelVoxel(x, y);
        var myEvent = new CustomEvent('voxel_selected_changed', {detail: {voxel: voxel, x: x, y: y}});
        window.dispatchEvent(myEvent);
    });

    async function set_voxel_selected({voxel, x, y}) {
        if (x === undefined)
            [x, y] = await getVoxelPixel({voxel});
        if (voxel === -1) {
            x = -1;
            y = -1;
        }
        selected_pos = [x, y];
        document.getElementsByName("x")[0].value = x;
        document.getElementsByName("y")[0].value = y;
        updatePointDisplay();
    }

    async function set_voxel_data(image) {
        const data32 = await voxels_to_flatmap(image);
        const processedImageData = new ImageData(data32, canvas.width, canvas.height);
        let ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.putImageData(processedImageData, 0, 0);

        set_texture(data32, canvas.width, canvas.height);
    }

    let last_overlay_matrix = null;
    async function plot_overlap_matrix(matrix){
        last_overlay_matrix = matrix
        let canvas = document.getElementById("matrix");
        let w = Math.sqrt(matrix.length);
        let ctx = canvas.getContext("2d");
        canvas.width = w;
        canvas.height = w;
        let data = new Uint8ClampedArray(matrix.length * 4);
        console.log(matrix)
        let max = Math.max(...matrix);
        for (let i = 0; i < matrix.length; i++) {
            let c= matrix[i] /max * 255
            data[i * 4] = c;
            data[i * 4 + 1] = c;
            data[i * 4 + 2] = c;
            data[i * 4 + 3] = 255;
        }

        const processedImageData = new ImageData(data, w, w);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.putImageData(processedImageData, 0, 0);
    }

    let canvas_matrix = document.getElementById("matrix");
    canvas_matrix.addEventListener("click", async function (event) {
        const rect = canvas_matrix.getBoundingClientRect();

        const form = document.getElementById("plotForm");
        const formData = new URLSearchParams(new FormData(form));
        const component_ids = formData.getAll("component_ids");

        // Calculate click position as percentage of bounding box dimensions
        const xPercent = ((event.clientX - rect.left) / rect.width);
        const yPercent = ((event.clientY - rect.top) / rect.height);
        let x = parseInt(xPercent * canvas_matrix.width);
        let y = parseInt(yPercent * canvas_matrix.height);

        document.getElementById("matrix_clicked").innerText = component_ids[x] + " " + component_ids[y] + " " + last_overlay_matrix[x * Math.sqrt(last_overlay_matrix.length) + y];

        //var myEvent = new CustomEvent('voxel_selected_changed', {detail: {voxel: -1}});
        //window.dispatchEvent(myEvent);

        var myEvent2 = new CustomEvent('display_components', {detail: {components: [component_ids[x], component_ids[y]]}});
        window.dispatchEvent(myEvent2);

        console.log(x, y)
    });

    return {
        set_voxel_data,
        set_voxel_selected,
        plot_overlap_matrix,
    }
}
