// worker.js
import {show_image, show_image2, get_components, get_count} from "./flat_map.mjs";

self.addEventListener('message', async function (e) {
    if (e.data.type === 'image') {
        const data32_index = await show_image(e.data);
        //const data32 = await voxels_to_flatmap(data32_index);

        // Post the result back to the main thread
        self.postMessage({type: 'image', data32_index});
    }
    if (e.data.type === 'image2') {
        const data32_index = await show_image2(e.data);

        // Post the result back to the main thread
        self.postMessage({type: 'image', data32_index});
    }
    if (e.data.type === 'pixel') {
        let pixel = await get_components(e.data);

        let counts = {};
        for (let index in e.data.component_ids) {
            let count = await get_count({component_id: e.data.component_ids_array[index], ...e.data});
            counts[e.data.component_ids[index]] = count;
        }

        // Post the result back to the main thread
        self.postMessage({pixel: pixel, type: 'pixel', counts: counts});
    }
});