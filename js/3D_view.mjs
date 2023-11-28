import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls";

import {
  loadNpy,
  cachedLoadNpy,
  getPngData,
  overlayImagesUint8,
} from "./numpy_to_js.mjs";
import { get_cmap, interpolateColor, get_cmap_display } from "./colormaps.mjs";

let animation_callback = { shape_change: null };

export async function initScene({ dom_elem }) {
  // if no element is defined, add a new one to the body
  if (!dom_elem) {
    dom_elem = document.createElement("canvas");
    document.body.appendChild(dom_elem);
  }
  let dom_controls = create_controls(dom_elem);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x151515);
  scene.dom_controls = dom_controls;

  const camera = new THREE.PerspectiveCamera(
    75,
    window.innerWidth / window.innerHeight,
    0.1,
    1000,
  );
  const renderer = new THREE.WebGLRenderer({ canvas: dom_controls.canvas });
  scene.renderer = renderer;
  scene.camera = camera;
  document.renderer = renderer;
  document.camera = camera;

  //camera.position.z = -1.7;
  camera.position.z = -2;
  //camera.position.z = -1.590;
  //camera.position.x = -0.601;

  // Set up orbit controls
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.update();
  scene.controls = controls;

  // Animation loop
  const light = new THREE.PointLight(0xffffff, 1, 100);
  light.position.set(10, 10, -10);
  scene.add(light);

  const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
  scene.add(ambientLight);

  function setLightStrength(strength, strength2) {
    ambientLight.intensity = strength;
    light.intensity = strength2;
  }

  window.setLightStrength = setLightStrength;
  scene.setLightStrength = setLightStrength;
  scene.initialized = false;

  const animate = (time) => {
    requestAnimationFrame(animate);

    for (let name in animation_callback) {
      if (animation_callback[name]) {
        animation_callback[name](time);
      }
    }

    controls.update();
    // link the light to the camera
    var campos = new THREE.Spherical().setFromVector3(camera.position);
    var lightpos = new THREE.Spherical(
      campos.radius,
      campos.phi,
      campos.theta + (30 / 180) * Math.PI,
    );
    light.position.setFromSpherical(lightpos);

    if (scene.initialized) {
      renderer.render(scene, camera);
    }
  };

  requestAnimationFrame(animate);

  function onWindowResize() {
    let container = dom_elem.parentElement;
    let width = container.clientWidth;
    let height = container.clientHeight - 4;

    // Update camera aspect ratio
    camera.aspect = width / height;
    camera.updateProjectionMatrix();

    // Update renderer size
    renderer.setSize(width, height);
  }
  scene.onWindowResize = onWindowResize;
  document.onWindowResize = onWindowResize;
  document.onWindowResize();

  // Add a resize event listener
  window.addEventListener("resize", onWindowResize, false);

  // Animation control variable
  var animationRequestID = null;

  function animateCamera(targetPos, lookAtTarget, duration) {
    controls.update();
    controls.enabled = false;
    //var targetPosition = new THREE.Vector3(0, 0, 2);
    //var lookAtTarget = new THREE.Vector3(0, 0, 0);
    //var duration = 2000; // 2000 ms

    // Cancel any ongoing animation
    if (animationRequestID) {
      cancelAnimationFrame(animationRequestID);
    }

    // Get initial position and orientation
    var initialPos = camera.position.clone();
    var initialSpherical = new THREE.Spherical().setFromVector3(initialPos);
    var targetSpherical = new THREE.Spherical().setFromVector3(targetPos);
    let phi = [initialSpherical.phi, targetSpherical.phi];
    let theta = [initialSpherical.theta, targetSpherical.theta];
    if (theta[1] - theta[0] > Math.PI) {
      theta[1] -= 2 * Math.PI;
    }
    if (theta[1] - theta[0] < -Math.PI) {
      theta[1] += 2 * Math.PI;
    }
    duration = Math.max(
      Math.abs(theta[1] - theta[0]) * 1000,
      Math.abs(phi[1] - phi[0]) * 1000,
    );

    // Get initial and target look-at position
    var initialLookAt = new THREE.Vector3(0, 0, -1).applyQuaternion(
      camera.quaternion,
    );
    var targetLookAt = lookAtTarget.clone().sub(initialPos).normalize();

    // Start time
    var startTime = Date.now();

    // Animation loop
    function animate() {
      // Calculate elapsed time and progress
      var elapsedTime = Date.now() - startTime;
      var progress = elapsedTime / duration;

      // Check if the animation is complete
      if (progress >= 1) {
        controls.reset();
        camera.position.copy(targetPos);
        camera.lookAt(lookAtTarget);
        renderer.render(scene, camera);
        controls.enabled = true;
        window.controls = controls;
        return;
      }

      // Interpolate spherical coordinates
      var interpolatedSpherical = new THREE.Spherical(
        THREE.MathUtils.lerp(
          initialSpherical.radius,
          targetSpherical.radius,
          progress,
        ),
        THREE.MathUtils.lerp(phi[0], phi[1], progress),
        THREE.MathUtils.lerp(theta[0], theta[1], progress),
      );

      // Interpolate look-at position
      var currentLookAt = new THREE.Vector3().lerpVectors(
        initialLookAt,
        targetLookAt,
        progress,
      );
      //console.log(interpolatedSpherical, currentLookAt)

      // Update camera position and orientation
      camera.position.setFromSpherical(interpolatedSpherical);
      //camera.lookAt(currentLookAt);
      camera.lookAt(lookAtTarget);

      // Render and request next frame
      //renderer.render(scene, camera);
      animationRequestID = requestAnimationFrame(animate);
    }

    // Start the animation
    animate();
  }

  window.animateCamera = animateCamera;
  window.THREE = THREE;

  function set_cmap_display(name = "turbo", color_count = 9) {
    get_cmap_display(dom_controls.colorbar, name, color_count);
  }
  scene.set_cmap_display = set_cmap_display;

  return scene;
}

function add_click(mesh, renderer, camera, callback) {
  function onDocumentMouseClick(event) {
    // Calculate mouse position in normalized device coordinates (-1 to +1)
    var canvasBounds = renderer.domElement.getBoundingClientRect();

    // Adjust for canvas position and normalize mouse position to -1 to +1 range
    mouse.x =
      ((event.clientX - canvasBounds.left) / canvasBounds.width) * 2 - 1;
    mouse.y =
      -((event.clientY - canvasBounds.top) / canvasBounds.height) * 2 + 1;

    // Update the picking ray with the camera and mouse position
    raycaster.setFromCamera(mouse, camera);

    // Calculate objects intersecting the picking ray
    var intersects = raycaster.intersectObject(mesh);

    if (intersects.length > 0) {
      // Get the first intersected object
      var intersection = intersects[0];

      // Get the closest vertex (you can also loop through vertices to find the closest one)
      var vertexIndex = intersection.face.a;
      callback(vertexIndex);
    } else {
      callback(-1);
    }
  }

  // Create a raycaster and a vector to hold the mouse click position
  var raycaster = new THREE.Raycaster();
  var mouse = new THREE.Vector2();

  // Add a click event listener to the canvas
  renderer.domElement.addEventListener("click", onDocumentMouseClick, false);
}

function addMesh(scene, pt, vtx) {
  // Define your points and vertices
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "color",
    new THREE.Float32BufferAttribute(new Float32Array(pt.shape[0] * 3), 3),
  );
  geometry.setIndex(vtx);
  geometry.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array(pt.data), 3),
  );

  geometry.addGroup(0, 606011 * 3, 0); // Use material index 0
  geometry.addGroup(606011 * 3, 655360 * 3, 1);
  //geometry.addGroup(0, 655360 * 3, 1);

  const textureLoader = new THREE.TextureLoader();
  const texture = textureLoader.load("static_data/foreground.png", () => {
    // Update rendering when the texture is loaded
    scene.renderer.render(scene, scene.camera);
  });
  texture.magFilter = THREE.NearestFilter;

  // MeshBasicMaterial
  const material = new THREE.MeshLambertMaterial({
    vertexColors: true,
    map: texture,
    side: THREE.DoubleSide,
  });
  texture.colorSpace = THREE.SRGBColorSpace;
  material.roi_texture = texture;

  const material2 = new THREE.MeshLambertMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
  });

  const mesh = new THREE.Mesh(geometry, [material, material2]);
  geometry.computeVertexNormals(); // This is important for diffuse shading
  scene.add(mesh);

  return mesh;
}

export async function add_brain({
  scene,
  pt_flat,
  faces_flat,
  pt_inflated,
  faces_inflated,
  pt_pia,
  faces_pia,
  pt_wm,
  faces_wm,
}) {
  let pt = await loadNpy(pt_flat);
  let vtx_flat = await loadNpy(faces_flat);
  vtx_flat = Array.prototype.slice.call(vtx_flat.data);
  let vtx = await loadNpy(faces_inflated);
  vtx = Array.prototype.slice.call(vtx.data);

  let pt2 = await loadNpy(pt_inflated);
  let pt3 = await loadNpy(pt_pia);
  let pt4 = await loadNpy(pt_wm);

  const curvature = (await cachedLoadNpy("static_data/curvature.npy")).data;

  let voxel_count = pt2.shape[0];

  const mesh = addMesh(scene, pt2, vtx);

  // Calculate scale and offset to normalize the UVs
  const scaleX = -0.15743183817062445; //*0 + 356/(1137.0*2);
  const scaleY = 0.349609375; //*0 + 356/(512.0*2);
  const offsetX = 0.5; //*0 + -1137.0/(1137.0*2);
  const offsetY = 0.5; //*0 + -512.0/(512.0*2);

  // Now map the vertices to UV space
  const uvs = new Float32Array(vtx_flat.length * 2);
  for (let i = 0; i < 327684; i++) {
    uvs[i * 2] = pt.data[i * 3] * scaleX + offsetX;
    uvs[i * 2 + 1] = pt.data[i * 3 + 1] * scaleY + offsetY;
  }
  window.uvs = uvs;
  window.vtx_flat = vtx_flat;
  window.pt = pt;
  mesh.geometry.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  //

  let last_shape_index = 1;
  let pivot = 0;

  function set_pivot(index) {
    pivot = index;
    set_shape(last_shape_index);
  }
  scene.dom_controls.slider_pivot_input.oninput = function () {
    set_pivot(this.value / 100);
  };

  function set_shape(index) {
    let flatness = 1 - Math.min(index, 1);
    setLightStrength(1 + 3 * flatness, (1 - flatness) * 5);

    function set_button_colors(i1, i2, i3, i4) {
      document.getElementsByClassName("img_flat")[0].style.backgroundColor =
        interpolateColor("#7d7d7d", "#9F2222", i1);
      document.getElementsByClassName("img_inflated")[0].style.backgroundColor =
        interpolateColor("#7d7d7d", "#9F2222", i2);
      document.getElementsByClassName("img_pia")[0].style.backgroundColor =
        interpolateColor("#7d7d7d", "#9F2222", i3);
      document.getElementsByClassName("img_wm")[0].style.backgroundColor =
        interpolateColor("#7d7d7d", "#9F2222", i4);
    }

    if (index < 1) {
      set_button_colors(1 - index, index, 0, 0);
    } else if (index < 2) {
      set_button_colors(0, 1 - (index - 1), index - 1, 0);
    } else {
      set_button_colors(0, 0, 1 - (index - 2), index - 2);
    }

    if (index === 0) {
      scene.controls.mouseButtons = {
        LEFT: THREE.MOUSE.PAN,
        MIDDLE: THREE.MOUSE.DOLLY,
        RIGHT: THREE.MOUSE.ROTATE,
      };
      scene.controls.touches = {
        ONE: THREE.TOUCH.PAN,
        TWO: THREE.TOUCH.DOLLY_PAN,
      };
    } else {
      scene.controls.mouseButtons = {
        LEFT: THREE.MOUSE.ROTATE,
        MIDDLE: THREE.MOUSE.DOLLY,
        RIGHT: THREE.MOUSE.PAN,
      };
      scene.controls.touches = {
        ONE: THREE.TOUCH.ROTATE,
        TWO: THREE.TOUCH.DOLLY_PAN,
      };
    }

    let p = pivot;
    if (index < 1) p = pivot * index;
    if (index < 1 && last_shape_index >= 1) mesh.geometry.setIndex(vtx_flat);
    if (index >= 1 && last_shape_index < 1) mesh.geometry.setIndex(vtx);
    last_shape_index = index;
    let pt_a = [pt, pt2, pt3, pt4][Math.floor(index)];
    let pt_b = [pt, pt2, pt3, pt4][Math.ceil(index)];
    let f = index % 1;
    let array = mesh.geometry.getAttribute("position").array;
    /*for (let i = 0; i < array.length; i += 1) {
            array[i] = pt_a.data[i]*(1-f) + pt_b.data[i] * f;
        }*/
    for (let i = 0; i < array.length; i += 3) {
      let point = [0, 0, 0];
      for (let j = 0; j < 3; j += 1)
        point[j] = pt_a.data[i + j] * (1 - f) + pt_b.data[i + j] * f;
      if (i >= array.length / 2) point = rotateAroundY(point, -90 * p, -1.2, p);
      //-0.00240851 -0.2672709   0.22112776
      else point = rotateAroundY(point, 90 * p, -1.2, p); //-0.00240851 -0.2672709   0.22112776
      for (let j = 0; j < 3; j += 1) array[i + j] = point[j];
    }
    if (index % 0.5 < 0.01) mesh.geometry.computeVertexNormals(); // This is important for diffuse shading
    mesh.geometry.getAttribute("position").needsUpdate = true;
    update_active_voxel();
  }

  function set_shape_animated(endIndex) {
    if (endIndex === 0) {
      animateCamera(
        new THREE.Vector3(0, 0, -2),
        new THREE.Vector3(0, 0, 0),
        2000,
      );
    } else {
      if (last_shape_index === 0) {
        animateCamera(
          new THREE.Vector3(-0.6, 0, -1.59),
          new THREE.Vector3(0, 0, 0),
          2000,
        );
      }
    }
    let duration = 1000;
    let startIndex = last_shape_index;

    animateShapeChange(duration, (progress) => {
      // Calculate the current index based on progress
      var currentIndex = startIndex + (endIndex - startIndex) * progress;

      // Call your set_shape function
      set_shape(currentIndex);
      document.getElementById("shape").value = currentIndex * 100;
    });
  }
  let i = 0;
  for (let elem of scene.dom_controls.slider_buttons.querySelectorAll(
    "button",
  )) {
    let value = i;
    elem.onclick = () => set_shape_animated(value);
    i += 1;
  }

  function rotateAroundY(point, angleDegrees, a, f) {
    // Convert angle to radians
    const theta = (angleDegrees * Math.PI) / 180.0;

    // Translate the point to the origin by subtracting (0, 0, a)
    const [x, y, z] = [point[0], point[1], point[2] - a];

    // Perform the rotation around the y-axis
    let xPrime = x * Math.cos(theta) + z * Math.sin(theta);
    let yPrime = y; // y-coordinate remains unchanged
    let zPrime = -x * Math.sin(theta) + z * Math.cos(theta);

    // Translate the point back to the original position by adding (0, 0, a)
    const finalX = xPrime;
    const finalY = yPrime;
    const finalZ = zPrime + a * (1 - f);

    return [finalX, finalY, finalZ];
  }

  document.animateShapeChange = set_shape_animated;
  document.set_pivot = set_pivot;
  document.set_shape = set_shape;
  scene.dom_controls.slider_shape_input.oninput = function () {
    set_shape(this.value / 100);
  };

  function set_mesh_colors(c) {
    mesh.geometry.getAttribute("color").array.set(c);
    mesh.geometry.getAttribute("color").needsUpdate = true;
  }

  function set_voxel_data(c, cmap_max) {
    console.time("set_voxel_data");
    let array = mesh.geometry.getAttribute("color").array;
    let my_cmap = get_cmap("turbo", cmap_max + 1);
    let cmap_gray = get_cmap("gray", 4);

    for (let i = 0; i < c.length; i += 1) {
      let clr;
      if (c[i] < 0) {
        if (curvature[i] > 0) {
          clr = cmap_gray.get_color(2);
        } else {
          clr = cmap_gray.get_color(1);
        }
      } else {
        clr = my_cmap.get_color(c[i]);
      }
      array.set(clr, i * 3);
    }
    mesh.geometry.getAttribute("color").needsUpdate = true;
    console.timeEnd("set_voxel_data");
  }
  set_voxel_data(
    new Float32Array(mesh.geometry.getAttribute("color").array.length / 3),
    8,
  );

  //---------
  const material2 = new THREE.MeshBasicMaterial({ color: 0xff0000 }); // greenish blue
  const cube = new THREE.Mesh(
    new THREE.BoxGeometry(0.01, 0.01, 0.01),
    material2,
  );
  cube.position.set(0, 0, 0.5);
  scene.add(cube);

  let lw = 0.005;
  const cube2 = new THREE.Mesh(new THREE.BoxGeometry(10, lw, lw), material2);
  cube2.position.set(0, 0, 0.5);
  scene.add(cube2);
  const cube3 = new THREE.Mesh(new THREE.BoxGeometry(lw, 10, lw), material2);
  cube3.position.set(0, 0, 0.5);
  scene.add(cube3);
  const cube4 = new THREE.Mesh(new THREE.BoxGeometry(lw, lw, 10), material2);
  cube4.position.set(0, 0, 0.5);
  scene.add(cube4);

  // Function to handle mouse click event
  add_click(mesh, scene.renderer, scene.camera, (e) => {
    var myEvent = new CustomEvent("voxel_selected_changed", {
      detail: { voxel: e },
    });
    window.dispatchEvent(myEvent);
  });

  function set_voxel_selected({ voxel }) {
    document.active_voxel = voxel;
    update_active_voxel();
  }

  function update_active_voxel() {
    let active_voxel = document.active_voxel;

    var [x, y, z] = [-1000, -1000, -1000];
    if (active_voxel !== -1) {
      let a = mesh.geometry.getAttribute("position").array;
      x = a[active_voxel * 3];
      y = a[active_voxel * 3 + 1];
      z = a[active_voxel * 3 + 2];
    }

    for (let obj of [cube, cube2, cube3, cube4]) {
      obj.position.set(x, y, z);
    }
    if (last_shape_index === 0) cube4.position.set(0, 0, -10000);
  }

  //

  let show_roi = true;

  function set_roi_show(show) {
    show_roi = show;
    set_texture(last_data[0], last_data[1], last_data[2]);
    return;
    if (show) mesh.material[0].map = mesh.material[0].roi_texture;
    else mesh.material[0].map = null;
    mesh.material[0].needsUpdate = true;
  }

  let last_data = null;

  async function set_texture(data, width, height) {
    if (typeof data === "promise") data = await data;
    width = width || data.shape[1];
    height = height || data.shape[0];

    last_data = [data, width, height];
    if (show_roi) {
      let foreground = await getPngData("static_data/foreground.png");
      data = overlayImagesUint8(data, foreground, width, height);
    }

    const texture = new THREE.DataTexture(
      data,
      width,
      height,
      THREE.RGBAFormat,
    );
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true; // Update the texture
    texture.flipY = true;
    texture.magFilter = THREE.NearestFilter;
    mesh.material[0].map = texture;
    mesh.material[0].needsUpdate = true;
    mesh.material[0].vertexColors = false;

    scene.initialized = true;
  }

  async function download_last_texture(filename = "downloadedImage.png", show_roi=false) {
    let [data, width, height] = last_data;
    if (show_roi) {
      let foreground = await getPngData("static_data/foreground.png");
      data = overlayImagesUint8(data, foreground, width, height);
    }
    let canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    var ctx = canvas.getContext("2d");
    var imgData = new ImageData(data, width, height);
    ctx.putImageData(imgData, 0, 0);

    // Step 3: Create Download Link
    const downloadLink = document.createElement("a");
    downloadLink.href = canvas.toDataURL();
    downloadLink.download = filename; // Suggest a filename for download

    // Append link to the body (optional, can be hidden)
    document.body.appendChild(downloadLink);

    // Step 4: Trigger the Download
    downloadLink.click();

    // Optional: Clean up
    document.body.removeChild(downloadLink);
  }

  window.set_texture = set_texture;
  window.download_last_texture = download_last_texture;

  set_shape(0);

  return {
    mesh,
    set_mesh_colors,
    set_pivot,
    set_shape,
    set_shape_animated,
    set_voxel_data,
    set_voxel_selected,
    set_roi_show,
    set_texture,
    download_last_texture,
  };
}

function animateShapeChange(duration, callback) {
  var startTime = null;

  function animate(time) {
    // Initialize the start time
    if (!startTime) {
      startTime = time;
    }

    // Calculate the progress (time elapsed) as a percentage of duration
    var progress = (time - startTime) / duration;

    // Clamp progress to [0, 1]
    progress = Math.min(1, progress);

    callback(progress);

    // Continue animation until progress reaches 1 (100%)
    if (progress >= 1) {
      animation_callback["shape_change"] = null;
    }
  }

  //animate();
  animation_callback["shape_change"] = animate;
}

function create_controls(parent) {
  let progress = document.createElement("div");
  progress.className = "progress";
  parent.appendChild(progress);

  let canvas = document.createElement("canvas");
  canvas.id = "plot3d";
  parent.appendChild(canvas);

  let slider_pivot = document.createElement("div");
  slider_pivot.className = "slidecontainer";
  slider_pivot.style.bottom = "70px";
  let slider_pivot_input = document.createElement("input");
  slider_pivot_input.type = "range";
  slider_pivot_input.className = "slider";
  slider_pivot_input.name = "pivot";
  slider_pivot_input.min = "0";
  slider_pivot_input.max = "100";
  slider_pivot_input.value = "0";
  slider_pivot_input.oninput = "document.set_pivot(this.value/100)";
  slider_pivot.appendChild(slider_pivot_input);
  parent.appendChild(slider_pivot);

  let slider_buttons = document.createElement("div");
  slider_buttons.className = "slider_buttons";
  parent.appendChild(slider_buttons);
  let names = ["img_flat", "img_inflated", "img_pia", "img_wm"];
  for (let i in names) {
    let button = document.createElement("button");
    button.className = "slider_button " + names[i];
    button.onclick = `brain_3d.set_shape_animated(i)`;
    slider_buttons.appendChild(button);
  }

  let slider_shape = document.createElement("div");
  slider_shape.className = "slidecontainer";
  let slider_shape_input = document.createElement("input");
  slider_shape_input.type = "range";
  slider_shape_input.className = "slider";
  slider_shape_input.id = "shape";
  slider_shape_input.name = "shape";
  slider_shape_input.min = "0";
  slider_shape_input.max = "300";
  slider_shape_input.value = "0";
  slider_shape_input.oninput = "document.set_shape(this.value/100)";
  slider_shape.appendChild(slider_shape_input);
  parent.appendChild(slider_shape);

  let colorbar = document.createElement("div");
  colorbar.className = "colorbar";
  let canvas_cbar = document.createElement("canvas");
  canvas_cbar.width = 200;
  canvas_cbar.height = 15;
  colorbar.appendChild(canvas_cbar);
  for (let i = 0; i < 5; i++) {
    colorbar.appendChild(document.createElement("span"));
  }
  parent.appendChild(colorbar);

  return {
    progress,
    canvas,
    slider_pivot,
    slider_pivot_input,
    slider_buttons,
    slider_shape,
    slider_shape_input,
    colorbar,
  };
}
