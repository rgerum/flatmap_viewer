function string_to_array(str) {
  if (!str) return undefined;
  str = str.trim();
  if (str === "") return undefined;
  return str.split(",").map((v) => parseInt(v));
}

function updateUrlParams(obj) {
  const paramsArray = [];

  for (const [key, values] of Object.entries(obj)) {
    console.log(key, values);
    if (key === "add_value") continue;
    if (typeof values === "object") {
      if(values.length > 0)
        paramsArray.push(`${key}=${values.join(",")}`);
    } else {
      paramsArray.push(`${key}=${values}`);
    }
  }

  const paramString = paramsArray.join("&");
  const newUrl = `${window.location.protocol}//${window.location.host}${window.location.pathname}?${paramString}`;
  history.pushState(null, "", newUrl);
}

function updateObjectFromUrl(obj) {
  const params = new URLSearchParams(window.location.search);
  for (const key in obj) {
    if (params.has(key)) {
      if (typeof obj[key] === "object")
        obj[key] = string_to_array(params.get(key));
      else obj[key] = params.get(key);
    }
  }
}

export function getUrlObject() {
  const params = new URLSearchParams(window.location.search);

  const handler = {
    set: function (obj, prop, value) {
      obj[prop] = value;
      if (prop === "add_value") return true;
      updateUrlParams(obj);
      return true;
    },
  };

  const myObject = new Proxy({}, handler);
  myObject.add_value = (key, value) => {
    console.log(typeof value, key, value);
    if (typeof value === "object")
      myObject[key] = string_to_array(params.get(key)) || value;
    else myObject[key] = params.get(key) || value;
  };

  window.addEventListener("popstate", function (event) {
    if (event.state) {
      for (const key in myObject) {
        if (event.state[key] !== undefined) {
          myObject[key] = event.state[key];
        }
      }
    } else {
      updateObjectFromUrl(myObject);
    }
  });

  // Example usage
  console.log(myObject); // Will contain URL parameters or default values
  //myObject.a = 10; // URL will update with the new value of 'a'
  return myObject;
}
