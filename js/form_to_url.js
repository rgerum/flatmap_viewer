function form_to_url(form) {
  const formData = new FormData(form);
  const groupedData = {};

  for (let input of document.getElementsByTagName("input")) {
    groupedData[input.name] = [];
  }

  formData.forEach((value, key) => {
    if (!groupedData[key]) {
      groupedData[key] = [];
    }
    groupedData[key].push(value);
  });

  const paramsArray = [];

  for (const [key, values] of Object.entries(groupedData)) {
    if (values.length > 1) {
      paramsArray.push(`${key}=${values.join(",")}`);
    } else if (values.length === 1) {
      paramsArray.push(`${key}=${values[0]}`);
    } else {
      paramsArray.push(`${key}=`);
    }
  }

  const paramString = paramsArray.join("&");
  const newUrl = `${window.location.protocol}//${window.location.host}${window.location.pathname}?${paramString}`;
  history.pushState(null, "", newUrl);
}

function link_forms() {
  document.querySelector("form").addEventListener("change", function () {
    form_to_url(this);
  });
}

function restore_form_from_url() {
  const searchParams = new URLSearchParams(window.location.search);

  const checkboxes = document.querySelectorAll('input[type="checkbox"]');
  checkboxes.forEach((checkbox) => {
    if (searchParams.has(checkbox.name)) {
      checkbox.checked = false;
    }
  });

  searchParams.forEach((value, key) => {
    const values = value.split(",");
    if (
      values.length === 1 &&
      document.querySelectorAll(`[name="${key}"]`).length === 1
    ) {
      let val = values[0];
      const input = document.querySelector(`[name="${key}"]`);
      if (input && (input.type === "checkbox" || input.type === "radio")) {
        input.checked = val != "";
      } else if (input) {
        input.value = val;
      }
    } else {
      values.forEach((val) => {
        const input = document.querySelector(`[name="${key}"][value="${val}"]`);
        if (input && (input.type === "checkbox" || input.type === "radio")) {
          input.checked = true;
        } else if (input) {
          input.value = val;
        }
      });
    }
  });
}

window.addEventListener("load", function () {
  link_forms();
});

window.addEventListener("popstate", function () {
  restore_form_from_url();
  document.update_plot();
});
