import { cmap } from "./colormaps.mjs";

function padNumber(num) {
  let numStr = num.toString();
  while (numStr.length < 3) {
    numStr = " " + numStr;
  }
  return numStr;
}

export function drawMatrix2(matrix, labels, callback) {
  const canvas = document.getElementById("matrixContainer");
  const current_children = {};

  for (let child of canvas.children) {
    current_children[child.id] = child;
  }
  let w = Math.sqrt(matrix.length);
  let max = Math.max(...matrix);
  let offset = 3;
  let label_size = 1.8;
  let cell_width = canvas.clientWidth / (w + offset);

  for (let i = 0; i < w; i++) {
    for (let j = 0; j < w; j++) {
      let id = padNumber(labels[i]) + padNumber(labels[j]);
      if (!current_children[id]) {
        let matrixCell = document.createElement("div");
        matrixCell.className = "matrixCell";
        matrixCell.id = id;
        matrixCell.title =
          labels[i] + " " + labels[j] + " " + matrix[i * w + j];
        canvas.appendChild(matrixCell);
        current_children[id] = matrixCell;
      }
      (function (x, y, i, j, v) {
        current_children[id].addEventListener("click", function () {
          callback(x, y, i, j, v);
        });
      })(i, j, labels[i], labels[j], matrix[i * w + j]);
      current_children[id].style.transform = `translate(${
        (j + offset) * cell_width
      }px, ${(i + offset) * cell_width}px) scale(${cell_width})`;
      current_children[id].style.backgroundColor = getColorForValue(
        matrix[i * w + j] / max,
      );
      delete current_children[id];
    }

    let id = "labelY_" + padNumber(labels[i]);
    if (!current_children[id]) {
      let matrixCell = document.createElement("div");
      matrixCell.className = "matrixCell";
      matrixCell.id = id;
      matrixCell.style.width = `${label_size}px`;
      matrixCell.innerText = padNumber(labels[i]);
      canvas.appendChild(matrixCell);
      current_children[id] = matrixCell;
    }
    current_children[id].style.transform = `translate(${
      (offset - label_size) * cell_width
    }px, ${(i + offset) * cell_width}px) scale(${cell_width})`;
    delete current_children[id];

    let id2 = "labelX_" + padNumber(labels[i]);
    if (!current_children[id2]) {
      let matrixCell = document.createElement("div");
      matrixCell.className = "matrixCell";
      matrixCell.id = id2;
      matrixCell.style.width = `${label_size}px`;
      matrixCell.innerText = padNumber(labels[i]);
      canvas.appendChild(matrixCell);
      current_children[id2] = matrixCell;
      console.log(id, matrixCell);
    }
    current_children[id2].style.transform = `translate(${
      (i + offset) * cell_width
    }px, ${
      (offset - label_size) * cell_width
    }px) scale(${cell_width}) rotate(-90deg)`;
    delete current_children[id2];
  }
  for (let id in current_children) {
    canvas.removeChild(current_children[id]);
  }
  console.log(current_children);
  return;
}
/*
drawMatrix2([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2])
 */

export function drawHeatmap(canvas, matrix, labels, vmax) {
  const ctx = canvas.getContext("2d");
  let w = Math.sqrt(matrix.length);

  const cellSize = 40; // Size of each cell in the heatmap
  const labelPadding = 65; // Increased space for rotated labels
  const fontSize = 30; // Font size for labels

  // Adjust canvas size
  canvas.width = labelPadding + cellSize * w;
  canvas.height = labelPadding + cellSize * w;

  ctx.font = `${fontSize}px Monospace`;

  for (let i = 0; i < w; i++) {
    for (let j = 0; j < w; j++) {
      ctx.fillStyle = getColorForValue((matrix[i * w + j] / vmax) * 255);
      ctx.fillRect(
        labelPadding + j * cellSize,
        labelPadding + i * cellSize,
        cellSize,
        cellSize,
      );

      ctx.fillStyle = "white";
      // Draw row labels
      if (j === 0) {
        const text = `${padNumber(labels[i])}`;
        const textWidth = ctx.measureText(text).width;
        ctx.fillText(
          text,
          labelPadding - textWidth - 5,
          labelPadding + i * cellSize + cellSize / 2 + fontSize / 3,
        );
      }
      // Draw column labels (rotated)
      if (i === 0) {
        const text = `${padNumber(labels[j])}`;
        ctx.save();
        ctx.translate(
          labelPadding + j * cellSize + cellSize / 2,
          labelPadding - 10,
        );
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(text, 0, 0);
        ctx.restore();
      }
    }
  }

  let click_event_to_x_y = (event) => {
    const rect = canvas.getBoundingClientRect();

    // Calculate click position as percentage of bounding box dimensions
    const xPercent = (event.clientX - rect.left) / rect.width;
    const yPercent = (event.clientY - rect.top) / rect.height;
    let x = parseInt(xPercent * canvas.width);
    let y = parseInt(yPercent * canvas.height);

    x -= labelPadding;
    y -= labelPadding;

    x = Math.floor(x / cellSize);
    y = Math.floor(y / cellSize);

    return [x, y];
  };
  return click_event_to_x_y;
}

const my_cmap = cmap["viridis"];
function getColorForValue(intensity) {
  let i = Math.round(Math.max(0, Math.min(intensity, 1)) * 255);
  let r = my_cmap[i * 3] * 255;
  let g = my_cmap[i * 3 + 1] * 255;
  let b = my_cmap[i * 3 + 2] * 255;
  // Implement color logic based on the value
  return `rgb(${r}, ${g}, ${b})`;
}
