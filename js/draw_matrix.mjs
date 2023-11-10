function padNumber(num) {
    let numStr = num.toString();
    while (numStr.length < 3) {
        numStr = ' ' + numStr;
    }
    return numStr;
}

export function drawHeatmap(canvas, matrix, labels, vmax) {
    const ctx = canvas.getContext('2d');
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
            ctx.fillStyle = getColorForValue(matrix[i * w + j] / vmax * 255);
            ctx.fillRect(labelPadding + j * cellSize, labelPadding + i * cellSize, cellSize, cellSize);

            ctx.fillStyle = "white";
            // Draw row labels
            if (j === 0) {
                const text = `${padNumber(labels[i])}`;
                const textWidth = ctx.measureText(text).width;
                ctx.fillText(text, labelPadding - textWidth - 5, labelPadding + i * cellSize + cellSize / 2 + fontSize / 3);
            }
            // Draw column labels (rotated)
            if (i === 0) {
                const text = `${padNumber(labels[j])}`;
                ctx.save();
                ctx.translate(labelPadding + j * cellSize + cellSize / 2, labelPadding - 10);
                ctx.rotate(-Math.PI / 2);
                ctx.fillText(text, 0, 0);
                ctx.restore();
            }
        }
    }

    let click_event_to_x_y = (event) => {
        const rect = canvas.getBoundingClientRect();

        // Calculate click position as percentage of bounding box dimensions
        const xPercent = ((event.clientX - rect.left) / rect.width);
        const yPercent = ((event.clientY - rect.top) / rect.height);
        let x = parseInt(xPercent * canvas.width);
        let y = parseInt(yPercent * canvas.height);

        x -= labelPadding;
        y -= labelPadding;

        x = Math.floor(x / cellSize);
        y = Math.floor(y / cellSize);

        return [x, y];
    }
    return click_event_to_x_y
}

function getColorForValue(intensity) {
    // Implement color logic based on the value
    return `rgb(${intensity}, ${intensity}, ${intensity})`;
}