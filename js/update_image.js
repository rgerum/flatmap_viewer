const worker = new Worker('js/worker.js');

worker.addEventListener('message', function(e) {
    if(e.data.type === 'image') {
        document.getElementById("spinner").style.display = "none";
        let canvas = document.getElementById("myCanvas");
        let ctx = canvas.getContext("2d");

        document.set_mesh_colors(e.data.data32_colors)

        const processedImageData = new ImageData(new Uint8ClampedArray(e.data.image.buffer), canvas.width, canvas.height);

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.putImageData(processedImageData, 0, 0);
        document.getElementById("spinner").style.display = "none";

    }
    if(e.data.type === 'image2') {
        document.getElementById("spinner").style.display = "none";
        let canvas = document.getElementById("myCanvas");
        let ctx = canvas.getContext("2d");

        const processedImageData = new ImageData(new Uint8ClampedArray(e.data.image.buffer), canvas.width, canvas.height);

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.putImageData(processedImageData, 0, 0);
        document.getElementById("spinner").style.display = "none";
    }
    if(e.data.type === 'pixel') {
        document.getElementById("clicked").innerText = "Clicked: ";
        let element_examples = document.getElementById("componentExamples");
        element_examples.innerHTML = "";
        for(let i of e.data.pixel) {
            add_row(element_examples, i);
            document.getElementById("clicked").innerText += " " + i + " (" + e.data.counts[i] + "), ";
        }
    }
});

async function startWorker(form_data) {
    document.getElementById("spinner").style.display = "block";

    // Start the worker with some data
    worker.postMessage({
        type: 'image',
        ...form_data
    });
}

async function startWorker2(list_component_ids_array, subject_ids, min_subject_overlap_count, layer_ids) {
    document.getElementById("spinner").style.display = "block";

    // Start the worker with some data
    worker.postMessage({
        type: 'image2',
        list_component_ids_array: list_component_ids_array,
        subject_ids: subject_ids,
        min_subject_overlap_count: min_subject_overlap_count
    });
}

async function getPixelValue(form_data) {
    let x = form_data.x;
    let y = form_data.y;
    let canvas = document.getElementById("myCanvasPoint");
    let ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.arc(x, y, 1, 0, 2 * Math.PI, false);
    ctx.rect(0, y, canvas.width, 1);
    ctx.rect(x, 0, 1, canvas.height);
    ctx.fillStyle = 'red';
    ctx.fill();

    // Start the worker with some data
    worker.postMessage({
        type: 'pixel',
        ...form_data
    });
}