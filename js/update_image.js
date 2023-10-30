const worker = new Worker('js/worker.js');

worker.addEventListener('message', function(e) {
    if(e.data.type === 'image') {
        document.getElementById("spinner").style.display = "none";
        let canvas = document.getElementById("myCanvas");
        let ctx = canvas.getContext("2d");

        const processedImageData = new ImageData(new Uint8ClampedArray(e.data.image.buffer), canvas.width, canvas.height);

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.putImageData(processedImageData, 0, 0);
        document.getElementById("spinner").style.display = "none";
    }
    if(e.data.type === 'pixel') {
        console.log(e.data)
        document.getElementById("clicked").innerText = "Clicked: ";
        let element_examples = document.getElementById("componentExamples");
        element_examples.innerHTML = "";
        for(let i of e.data.pixel) {
            add_row(element_examples, i);
            document.getElementById("clicked").innerText += " " + i + " (" + e.data.counts[i] + "), ";
        }
    }
});

async function startWorker(component_ids_array, subject_ids, min_subject_overlap_count) {
    document.getElementById("spinner").style.display = "block";

    // Start the worker with some data
    worker.postMessage({
        type: 'image',
        component_ids_array: component_ids_array,
        subject_ids: subject_ids,
        min_subject_overlap_count: min_subject_overlap_count
    });
}

async function getPixelValue(component_ids_array, component_ids, subject_ids, min_subject_overlap_count, x, y) {
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
        component_ids_array: component_ids_array,
        component_ids: component_ids,
        subject_ids: subject_ids,
        min_subject_overlap_count: min_subject_overlap_count, x: x, y: y
    });
}