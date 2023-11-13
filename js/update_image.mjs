import {overlap_matrix} from "./flat_map.mjs";

const worker = new Worker('js/worker.mjs', {type: 'module'});

let update_id = 0;
worker.addEventListener('message', function (e) {
    if (e.data.type === 'image') {
        var myEvent = new CustomEvent('voxel_data_changed',
            {
                detail:
                    {
                        image: e.data.data32_index, ...e.data
                    }
            });
        window.dispatchEvent(myEvent);

        console.log("update_id", e.data.update_id, update_id)
        if(e.data.update_id === update_id)
            document.querySelectorAll(".spinner").forEach(x => x.style.display = "none");
    }
    if (e.data.type === 'pixel') {
        var myEvent = new CustomEvent('display_components', {detail: {components: e.data.pixel}});
        window.dispatchEvent(myEvent);

        document.getElementById("clicked").innerText = "Clicked: ";
        document.getElementsByName("x")[0].value = e.data.x;
        document.getElementsByName("y")[0].value = e.data.y;
        for (let i of e.data.pixel) {
            document.getElementById("clicked").innerText += " " + i + " (" + e.data.counts[i] + "), ";
        }
    }
});

export async function startWorker(form_data) {
    document.querySelectorAll(".spinner").forEach(x => x.style.display = "block");

    // Start the worker with some data
    worker.postMessage({
        type: 'image',
        update_id: ++update_id,
        ...form_data
    });
}

export async function startWorker2(form_data) {
    document.querySelectorAll(".spinner").forEach(x => x.style.display = "block");

    // Start the worker with some data
    worker.postMessage({
        type: 'image2',
        ...form_data
    });
}


export async function getPixelValue(form_data) {
    // Start the worker with some data
    worker.postMessage({
        type: 'pixel',
        ...form_data
    });
}