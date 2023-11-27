export function add_image(parent, src, w=128, h=128) {
  let img = document.createElement("img");
  img.src = src;
  if (w) img.width = w;
  if (h) img.height = h;
  parent.appendChild(img);

  return img;
}

export function create_row(parent, text) {
  let row = document.createElement("div");

  let title = document.createElement("div");
  title.className = "row_title2";
  let title2 = document.createElement("div");
  title2.innerText = text;
  title.appendChild(title2);
  row.appendChild(title);

  parent.appendChild(row);
  return row;
}
