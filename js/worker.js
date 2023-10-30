// worker.js
importScripts('numpy_to_js.js');
importScripts('flat_map.js');

self.addEventListener('message', async function(e) {
    if(e.data.type === 'image') {
         const outputData = await show_image(e.data.component_ids_array, e.data.subject_ids, e.data.min_subject_overlap_count);

        // Post the result back to the main thread
        self.postMessage({image: outputData, type: 'image'});
    }
    if(e.data.type === 'pixel') {
        console.log(e.data)
        let pixel = await get_components(e.data.component_ids_array, e.data.component_ids, e.data.subject_ids, e.data.min_subject_overlap_count, e.data.x, e.data.y);

        let counts = {};
        for(let index in e.data.component_ids) {
            let count = await get_count(e.data.component_ids_array[index], e.data.subject_ids, e.data.min_subject_overlap_count);
            counts[e.data.component_ids[index]] = count;
        }

        // Post the result back to the main thread
        self.postMessage({pixel: pixel, type: 'pixel', counts: counts});
    }
});