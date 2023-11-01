// worker.js
importScripts('numpy_to_js.js');
importScripts('flat_map.js');

self.addEventListener('message', async function(e) {
    if(e.data.type === 'image') {
         const [data32, data32_colors] = await show_image(e.data);

        // Post the result back to the main thread
        self.postMessage({image: data32, type: 'image', data32_colors});
    }
    if(e.data.type === 'image2') {
         const outputData = await show_image2(e.data.list_component_ids_array, e.data.subject_ids, e.data.min_subject_overlap_count, e.data.layer_ids);

        // Post the result back to the main thread
        self.postMessage({image: outputData, type: 'image2'});
    }
    if(e.data.type === 'pixel') {
        let pixel = await get_components(e.data);

        let counts = {};
        for(let index in e.data.component_ids) {
            let count = await get_count({component_id: e.data.component_ids_array[index], ...e.data});
            counts[e.data.component_ids[index]] = count;
        }

        // Post the result back to the main thread
        self.postMessage({pixel: pixel, type: 'pixel', counts: counts});
    }
});