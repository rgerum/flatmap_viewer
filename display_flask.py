from flask import Flask, Response, render_template, request, jsonify, send_from_directory
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import time
from flask_compress import Compress


from plot_clean import  all_component_ids, subject_ids

app = Flask(__name__)
app.config["COMPRESS_REGISTER"] = False  # disable default compression of all eligible requests
compress = Compress()
compress.init_app(app)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/init_data')
def init_data():
    print(all_component_ids)
    return jsonify({"all_component_ids": all_component_ids, "subject_ids": subject_ids})


@app.route('/process_coordinates', methods=['POST'])
def process_coordinates():
    # Get the percentage coordinates from the request data
    x_percent = request.json.get('x_percent', 0)
    y_percent = request.json.get('y_percent', 0)

    x, y, components, components_all = plotter.get_components(x_percent, y_percent)

    return jsonify({"message": "Coordinates received successfully!", "x": x, "y": y,
                    "components": components, "components_all": components_all})


# The path to your custom static files folder
MY_STATIC_PATH = "cache_top_image"


@app.route('/array.npy')
@compress.compressed()
def serve_file():
    return send_from_directory('.', 'array.bin')#, as_attachment=False, mimetype='application/octet-stream')


@app.route('/array.npy.gz')
def serve_file2():
    with open(f"array.bin.gz", "rb") as f:
        content = f.read()

    response = Response(content)
    response.headers['Content-Encoding'] = 'gzip'
    response.headers['Content-Type'] = 'application/octet-stream'

    return response

    return send_from_directory('.', 'array.bin.gz', as_attachment=False, mimetype='application/octet-stream')

@app.route('/cache_masks/<path:filename>')
def serve_file2_folder(filename):
    with open(f"cache_masks/{filename}", "rb") as f:
        content = f.read()

    response = Response(content)
    response.headers['Content-Encoding'] = 'gzip'
    response.headers['Content-Type'] = 'application/octet-stream'

    return response

    return send_from_directory('.', 'array.bin.gz', as_attachment=False, mimetype='application/octet-stream')



@app.route('/static_data/<path:filename>')
def serve_static(filename):
    return send_from_directory("static_data", filename)


@app.route('/js/<path:filename>')
def serve_static_js(filename):
    return send_from_directory("js", filename)

@app.route('/img/<path:filename>')
def serve_static_img(filename):
    return send_from_directory("img", filename)



@app.route('/plot')
def plotting():
    subject_ids = [int(i) for i in request.args.getlist('subject_ids')]
    component_ids = [int(i) for i in request.args.getlist('component_ids')]
    min_subject_overlap_count = int(request.args.get('min_subject_overlap_count', 1))
    print(subject_ids, component_ids, min_subject_overlap_count)

    im = plotter.plot(component_ids, subject_ids, min_subject_overlap_count)

    t = time.time()
    buf = io.BytesIO()
    plt.imsave(buf, im, format="png", cmap="turbo")
    buf.seek(0)
    print("time save", time.time() - t)
    return Response(buf, content_type="image/png")


@app.route('/plot0')
def plotting0():
    buf = io.BytesIO()
    plt.imsave(buf, plotter.im1, format="png")
    buf.seek(0)
    return Response(buf, content_type="image/png")

@app.route('/plot2')
def plotting2():
    buf = io.BytesIO()
    plt.imsave(buf, plotter.im2, format="png")
    buf.seek(0)
    return Response(buf, content_type="image/png")

if __name__ == '__main__':
    app.run(debug=True)