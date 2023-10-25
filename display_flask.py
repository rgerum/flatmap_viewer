from flask import Flask, Response, render_template, request, jsonify
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import io
import time

from plot_clean import Plotter, all_component_ids, subject_ids

app = Flask(__name__)
plotter = Plotter()

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

    components, components_all = plotter.get_components(x_percent, y_percent)

    return jsonify({"message": "Coordinates received successfully!", "x_percent": x_percent, "y_percent": y_percent,
                    "components": components, "components_all": components_all})


@app.route('/plot')
def plotting():
    subject_ids = [int(i) for i in request.args.getlist('subject_ids')]
    component_ids = [int(i) for i in request.args.getlist('component_ids')]
    min_subject_overlap_count = int(request.args.get('min_subject_overlap_count'))
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