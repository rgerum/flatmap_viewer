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
         const [data32, data32_colors] = await show_image2(e.data);

        // Post the result back to the main thread
        self.postMessage({image: data32, type: 'image2', data32_colors});
    }
    if(e.data.type === 'pixel') {
        [e.data.x, e.data.y] = await getVoxelPixel(e.data);
        let pixel = await get_components(e.data);

        let counts = {};
        for(let index in e.data.component_ids) {
            let count = await get_count({component_id: e.data.component_ids_array[index], ...e.data});
            counts[e.data.component_ids[index]] = count;
        }

        // Post the result back to the main thread
        self.postMessage({pixel: pixel, type: 'pixel', counts: counts, x: e.data.x, y: e.data.y});
    }
});