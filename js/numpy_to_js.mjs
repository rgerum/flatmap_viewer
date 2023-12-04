export async function loadNpy(url) {
  const response = await fetch(url);
  const arrayBuffer = await response.arrayBuffer();
  const dataView = new DataView(arrayBuffer);

  // Check magic number
  const magic = Array.from(new Uint8Array(arrayBuffer.slice(0, 6)))
    .map((byte) => String.fromCharCode(byte))
    .join("");
  if (magic !== "\x93NUMPY") {
    throw new Error(`Not a .npy file filename: ${url}`);
  }

  // Parse header
  const headerLength = dataView.getUint16(8, true); // Little endian
  const headerStr = new TextDecoder().decode(
    arrayBuffer.slice(10, 10 + headerLength),
  );
  const header = eval(
    "(" + headerStr.toLowerCase().replace("(", "[").replace(")", "]") + ")",
  );
  const dtype = header.descr;
  const shape = header.shape;

  let data;
  if (dtype === "|u1" || dtype === "|b1") {
    data = new Uint8Array(arrayBuffer, 10 + headerLength);
  } else if (dtype === "<f4") {
    data = new Float32Array(arrayBuffer, 10 + headerLength);
  } else if (dtype === "<i4") {
    data = new Int32Array(arrayBuffer, 10 + headerLength);
  } else {
    throw new Error("Unsupported dtype. Only Uint8 is supported. Got" + dtype);
  }

  return {
    dtype: dtype,
    shape: shape,
    data: data,
  };
}

export async function getPngData(url) {
  return new Promise((resolve, reject) => {
    // Create an image element
    const img = new Image();
    img.crossOrigin = "Anonymous"; // This is important for loading images from other domains

    // Set up what to do after the image has loaded
    img.onload = () => {
      // Create a canvas with the same dimensions as the image
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      canvas.width = img.width;
      canvas.height = img.height;

      // Draw the image onto the canvas
      ctx.drawImage(img, 0, 0);

      // Retrieve the pixel data from the canvas
      const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const pixelData = imageData.data; // Uint8ClampedArray of pixel data
      pixelData.shape = [canvas.height, canvas.width, 4];

      // Resolve the promise with the pixel data
      resolve(pixelData);
    };

    // Reject the promise if there's an error loading the image
    img.onerror = () => {
      reject(new Error(`Failed to load image at ${url}`));
    };

    // Start loading the image
    img.src = url;
  });
}

export function overlayImagesUint8(baseArray, overlayArray, width, height) {
  let width2 = overlayArray?.shape[1] || width;
  let height2 = overlayArray?.shape[0] || height;

  // Create a new array to hold the result, four entries for each pixel (RGBA)
  let resultArray = new Uint8ClampedArray(width * height * 4);

  for (let i = 0; i < width * height * 4; i += 4) {
    // Extract RGBA components for the base image
    let baseR = baseArray[i];
    let baseG = baseArray[i + 1];
    let baseB = baseArray[i + 2];
    let baseA = baseArray[i + 3];

    let j =
      Math.floor(i / (width * 4)) * (width2 * 4) +
      Math.floor((i % (width * 4)) / 4) * 4 +
      (i % 4);

    // Extract RGBA components for the overlay image
    let overlayR = overlayArray[j];
    let overlayG = overlayArray[j + 1];
    let overlayB = overlayArray[j + 2];
    let overlayA = overlayArray[j + 3];

    // Normalize the alpha values to the range 0-1
    baseA /= 255;
    overlayA /= 255;

    // Compute the combined alpha channel
    let combinedA = overlayA + baseA * (1 - overlayA);

    // Alpha blend the overlay color and base color
    let r = overlayR * overlayA + baseR * baseA * (1 - overlayA); // / combinedA;
    let g = overlayG * overlayA + baseG * baseA * (1 - overlayA); // / combinedA;
    let b = overlayB * overlayA + baseB * baseA * (1 - overlayA); // / combinedA;

    // Assign blended values to result array
    resultArray[i] = r; // * 255;
    resultArray[i + 1] = g; // * 255;
    resultArray[i + 2] = b; // * 255;
    resultArray[i + 3] = combinedA * 255;
  }

  return resultArray;
}

const cached = {};

export async function cachedLoadNpy(url) {
  if (url in cached) {
    return cached[url];
  }
  const data = await loadNpy(url);
  cached[url] = data;
  return data;
}


export async function loadAllPromises(promises) {
  let result = {};
  let promise_list = [];
  for(let key in promises) {
    let current_key = key;
    function  set(data) {
      console.log("loaded", current_key);
      result[current_key] = data;
    }
    promises[key].then(set);//data => result[current_key] = data
    promise_list.push(promises[key]);
  }
  await Promise.all(promise_list);
  console.log("loaded", result);
  return result;
}