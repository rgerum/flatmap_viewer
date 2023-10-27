async function loadNpy(url) {
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    const dataView = new DataView(arrayBuffer);

    // Check magic number
    const magic = Array.from(new Uint8Array(arrayBuffer.slice(0, 6)))
        .map(byte => String.fromCharCode(byte))
        .join('');
    if (magic !== '\x93NUMPY') {
        throw new Error('Not a .npy file');
    }

    // Parse header
    const headerLength = dataView.getUint16(8, true);  // Little endian
    const headerStr = new TextDecoder().decode(arrayBuffer.slice(10, 10 + headerLength));
    const header = eval("(" + headerStr.toLowerCase().replace('(', '[').replace(')', ']') + ")");
    const dtype = header.descr;
    const shape = header.shape;

    if (dtype !== '|u1' && dtype !== '|b1') {
        throw new Error('Unsupported dtype. Only Uint8 is supported.', dtype);
    }

    // Extract the data as Uint8Array
    const data = new Uint8Array(arrayBuffer, 10 + headerLength);

    return {
        dtype: dtype,
        shape: shape,
        data: data
    };
}