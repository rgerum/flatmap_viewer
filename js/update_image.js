const worker = new Worker('js/worker.js');

worker.addEventListener('message', function(e) {
    if(e.data.type === 'image') {
        console.log("voxel_data_changed event")
        var myEvent = new CustomEvent('voxel_data_changed',  { detail: {image: e.data.data32_index} });
        window.dispatchEvent(myEvent);

        /*
        let canvas = document.getElementById("myCanvas");
        let ctx = canvas.getContext("2d");

        //document.set_mesh_colors(e.data.data32_colors)
        window.brain_3d.set_voxel_data(e.data.data32_index);

        const processedImageData = new ImageData(new Uint8ClampedArray(e.data.image.buffer), canvas.width, canvas.height);

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.putImageData(processedImageData, 0, 0);
        */
        document.querySelectorAll(".spinner").forEach(x => x.style.display = "none");

    }
    if(e.data.type === 'image2') {
        let canvas = document.getElementById("myCanvas");
        let ctx = canvas.getContext("2d");

        document.set_mesh_colors(e.data.data32_colors);

        const processedImageData = new ImageData(new Uint8ClampedArray(e.data.image.buffer), canvas.width, canvas.height);

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.putImageData(processedImageData, 0, 0);
        document.querySelectorAll(".spinner").forEach(x => x.style.display = "block");
    }
    if(e.data.type === 'pixel') {
        document.getElementById("clicked").innerText = "Clicked: ";
        let element_examples = document.getElementById("componentExamples");
        element_examples.innerHTML = "";
        document.getElementsByName("x")[0].value = e.data.x;
        document.getElementsByName("y")[0].value = e.data.y;
        for(let i of e.data.pixel) {
            add_row(element_examples, i);
            document.getElementById("clicked").innerText += " " + i + " (" + e.data.counts[i] + "), ";
        }
    }
});

async function startWorker(form_data) {
    document.querySelectorAll(".spinner").forEach(x => x.style.display = "block");

    // Start the worker with some data
    worker.postMessage({
        type: 'image',
        ...form_data
    });
}

async function startWorker2(form_data) {
    document.querySelectorAll(".spinner").forEach(x => x.style.display = "block");

    // Start the worker with some data
    worker.postMessage({
        type: 'image2',
        ...form_data
    });
}



async function getPixelValue(form_data) {
    // Start the worker with some data
    worker.postMessage({
        type: 'pixel',
        ...form_data
    });
}