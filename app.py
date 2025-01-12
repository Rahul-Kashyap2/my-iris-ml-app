import os
import joblib
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from flask import Flask, request, jsonify
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

app = Flask(__name__)

# Load the binary models (trained via train.py)
model_bin1 = joblib.load(os.path.join("models", "model_binary1.pkl"))  # Setosa vs Not
model_bin2 = joblib.load(os.path.join("models", "model_binary2.pkl"))  # Versicolor vs Virginica

# We'll also load the full Iris dataset for on-demand clustering
iris_data = load_iris()
X_iris = iris_data.data  # shape: (150, 4)


@app.route('/')
def home():
    """
    Main homepage with Apple-like design and links to sub-pages:
    - Binary1: Setosa vs. Not
    - Binary2: Versicolor vs. Virginica
    - Clustering: K-Means with user input.
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Iris ML Experiments</title>
        <link rel="stylesheet"
              href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/css/bootstrap.min.css">
        <style>
            body {
                background: #f5f5f7;
                color: #1d1d1f;
                font-family: -apple-system, BlinkMacSystemFont,
                             'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                margin: 0;
                padding: 0;
            }
            .navbar {
                background-color: #000;
            }
            .navbar-brand, .nav-link {
                color: #fff !important;
            }
            .hero {
                text-align: center;
                padding: 60px 20px;
            }
            .hero h1 {
                font-size: 3rem;
                font-weight: 600;
                margin-bottom: 20px;
            }
            .hero p {
                font-size: 1.2rem;
                color: #555;
            }
            .buttons {
                margin-top: 40px;
            }
            .btn-apple {
                background-color: #0070c9;
                color: white;
                border: none;
                padding: 15px 30px;
                font-size: 1rem;
                border-radius: 50px;
                transition: background-color 0.3s ease, transform 0.2s;
                text-decoration: none;
                display: inline-flex;
                align-items: center;
                margin: 10px;
            }
            .btn-apple i {
                margin-right: 10px;
                font-size: 1.2rem;
            }
            .btn-apple:hover {
                background-color: #005ea6;
                transform: translateY(-2px);
            }
            footer {
                text-align: center;
                padding: 20px;
                background: #eaeaea;
                margin-top: 40px;
            }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg">
            <a class="navbar-brand" href="#">Iris ML Experiments</a>
        </nav>

        <div class="hero">
            <h1>Welcome to Iris ML Experiments</h1>
            <p>Try out our binary classifiers or create your own clusters!</p>
            <div class="buttons">
                <a href="/binary1" class="btn-apple">
                  <i class="fas fa-seedling"></i> Setosa vs. Not Setosa
                </a>
                <a href="/binary2" class="btn-apple">
                  <i class="fas fa-leaf"></i> Versicolor vs. Virginica
                </a>
                <a href="/clustering" class="btn-apple">
                  <i class="fas fa-chart-pie"></i> Clustering
                </a>
            </div>
        </div>

        <footer>
            <p>Designed with an Apple-inspired flair.</p>
            <!-- Font Awesome for icons (optional) -->
            <script src="https://kit.fontawesome.com/your_font_awesome_kit.js" crossorigin="anonymous"></script>
        </footer>
    </body>
    </html>
    """


###############################################################################
# BINARY 1: Setosa vs. Not Setosa
###############################################################################
@app.route('/binary1')
def binary1_page():
    """
    Page that lets the user input the 4 features, then calls /predict_binary1
    to determine if it's Setosa or Not Setosa.
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Setosa vs Not Setosa</title>
        <link rel="stylesheet"
              href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/css/bootstrap.min.css">
        <style>
            body {
                background: #f5f5f7;
                color: #1d1d1f;
                font-family: -apple-system, BlinkMacSystemFont,
                             'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            }
            .navbar {
                background-color: #000;
            }
            .navbar-brand, .nav-link {
                color: #fff !important;
            }
            .hero {
                text-align: center;
                padding: 30px 20px;
            }
            .hero h1 {
                font-size: 2rem;
                font-weight: 600;
                margin-bottom: 10px;
            }
            .predict-form {
                max-width: 500px;
                margin: 0 auto;
                margin-top: 20px;
            }
            .btn-apple {
                background-color: #0070c9;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 1rem;
                border-radius: 50px;
                transition: background-color 0.3s ease, transform 0.2s;
            }
            .btn-apple:hover {
                background-color: #005ea6;
                transform: translateY(-2px);
            }
            .result-box {
                margin-top: 20px;
                font-size: 1.2rem;
                font-weight: 500;
            }
            a.btn-apple {
                text-decoration: none;
                margin-top: 20px;
                display: inline-block;
            }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg">
            <a class="navbar-brand" href="/">Iris ML Experiments</a>
        </nav>

        <div class="hero">
            <h1>Setosa vs. Not Setosa</h1>
        </div>

        <div class="container">
            <form class="predict-form">
                <div class="form-group">
                    <label for="sepal_length">Sepal Length</label>
                    <input type="number" step="any" class="form-control" id="sepal_length" required />
                </div>
                <div class="form-group">
                    <label for="sepal_width">Sepal Width</label>
                    <input type="number" step="any" class="form-control" id="sepal_width" required />
                </div>
                <div class="form-group">
                    <label for="petal_length">Petal Length</label>
                    <input type="number" step="any" class="form-control" id="petal_length" required />
                </div>
                <div class="form-group">
                    <label for="petal_width">Petal Width</label>
                    <input type="number" step="any" class="form-control" id="petal_width" required />
                </div>
                <button type="submit" class="btn-apple">Predict</button>
            </form>

            <div id="result" class="result-box"></div>
            <a href="/" class="btn-apple">Go Back Home</a>
        </div>

        <script>
        const form = document.querySelector('.predict-form');
        const resultDiv = document.getElementById('result');

        form.addEventListener('submit', async (event) => {
            event.preventDefault();

            const sl = document.getElementById('sepal_length').value;
            const sw = document.getElementById('sepal_width').value;
            const pl = document.getElementById('petal_length').value;
            const pw = document.getElementById('petal_width').value;

            const payload = {
                sepal_length: sl,
                sepal_width: sw,
                petal_length: pl,
                petal_width: pw
            };

            try {
                const response = await fetch('/predict_binary1', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();

                if(data.error){
                    resultDiv.innerHTML = `
                        <div class="alert alert-danger" role="alert">
                            ${data.error}
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = `
                        <div class="alert alert-success" role="alert">
                            Prediction: <strong>${data.prediction}</strong>
                        </div>
                    `;
                }
            } catch(err) {
                console.error(err);
                resultDiv.innerHTML = `
                    <div class="alert alert-danger" role="alert">
                        An error occurred.
                    </div>
                `;
            }
        });
        </script>
    </body>
    </html>
    """


@app.route('/predict_binary1', methods=['POST'])
def predict_binary1():
    """
    Endpoint: Reads the 4 features, uses model_bin1, returns "Setosa" or "Not Setosa".
    """
    try:
        data = request.get_json(force=True)
        sl = float(data['sepal_length'])
        sw = float(data['sepal_width'])
        pl = float(data['petal_length'])
        pw = float(data['petal_width'])

        features = np.array([sl, sw, pl, pw]).reshape(1, -1)
        pred = model_bin1.predict(features)[0]  # 1 => setosa, 0 => not

        return jsonify({"prediction": "Setosa" if pred == 1 else "Not Setosa"})
    except:
        return jsonify({"error": "Invalid Input"}), 400


###############################################################################
# BINARY 2: Versicolor vs. Virginica
###############################################################################
@app.route('/binary2')
def binary2_page():
    """
    Page for predicting if a flower is Versicolor or Virginica.
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Versicolor vs. Virginica</title>
        <link rel="stylesheet"
              href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/css/bootstrap.min.css">
        <style>
            body {
                background: #f5f5f7;
                color: #1d1d1f;
                font-family: -apple-system, BlinkMacSystemFont,
                             'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            }
            .navbar {
                background-color: #000;
            }
            .navbar-brand, .nav-link {
                color: #fff !important;
            }
            .hero {
                text-align: center;
                padding: 30px 20px;
            }
            .hero h1 {
                font-size: 2rem;
                font-weight: 600;
                margin-bottom: 10px;
            }
            .predict-form {
                max-width: 500px;
                margin: 0 auto;
                margin-top: 20px;
            }
            .btn-apple {
                background-color: #0070c9;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 1rem;
                border-radius: 50px;
                transition: background-color 0.3s ease, transform 0.2s;
            }
            .btn-apple:hover {
                background-color: #005ea6;
                transform: translateY(-2px);
            }
            .result-box {
                margin-top: 20px;
                font-size: 1.2rem;
                font-weight: 500;
            }
            a.btn-apple {
                text-decoration: none;
                margin-top: 20px;
                display: inline-block;
            }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg">
            <a class="navbar-brand" href="/">Iris ML Experiments</a>
        </nav>

        <div class="hero">
            <h1>Versicolor vs. Virginica</h1>
        </div>

        <div class="container">
            <form class="predict-form">
                <div class="form-group">
                    <label for="sepal_length">Sepal Length</label>
                    <input type="number" step="any" class="form-control" id="sepal_length" required />
                </div>
                <div class="form-group">
                    <label for="sepal_width">Sepal Width</label>
                    <input type="number" step="any" class="form-control" id="sepal_width" required />
                </div>
                <div class="form-group">
                    <label for="petal_length">Petal Length</label>
                    <input type="number" step="any" class="form-control" id="petal_length" required />
                </div>
                <div class="form-group">
                    <label for="petal_width">Petal Width</label>
                    <input type="number" step="any" class="form-control" id="petal_width" required />
                </div>
                <button type="submit" class="btn-apple">Predict</button>
            </form>

            <div id="result" class="result-box"></div>
            <a href="/" class="btn-apple">Go Back Home</a>
        </div>

        <script>
        const form = document.querySelector('.predict-form');
        const resultDiv = document.getElementById('result');

        form.addEventListener('submit', async (event) => {
            event.preventDefault();

            const sl = document.getElementById('sepal_length').value;
            const sw = document.getElementById('sepal_width').value;
            const pl = document.getElementById('petal_length').value;
            const pw = document.getElementById('petal_width').value;

            const payload = {
                sepal_length: sl,
                sepal_width: sw,
                petal_length: pl,
                petal_width: pw
            };

            try {
                const response = await fetch('/predict_binary2', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const data = await response.json();

                if(data.error){
                    resultDiv.innerHTML = `
                        <div class="alert alert-danger" role="alert">
                            ${data.error}
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = `
                        <div class="alert alert-success" role="alert">
                            Prediction: <strong>${data.prediction}</strong>
                        </div>
                    `;
                }
            } catch(err) {
                console.error(err);
                resultDiv.innerHTML = `
                    <div class="alert alert-danger" role="alert">
                        An error occurred.
                    </div>
                `;
            }
        });
        </script>
    </body>
    </html>
    """


@app.route('/predict_binary2', methods=['POST'])
def predict_binary2():
    """
    Endpoint: Reads features, uses model_bin2, returns "Versicolor" or "Virginica".
    """
    try:
        data = request.get_json(force=True)
        sl = float(data['sepal_length'])
        sw = float(data['sepal_width'])
        pl = float(data['petal_length'])
        pw = float(data['petal_width'])

        features = np.array([sl, sw, pl, pw]).reshape(1, -1)
        pred = model_bin2.predict(features)[0]  # 0 => Versicolor, 1 => Virginica
        result = "Versicolor" if pred == 0 else "Virginica"
        return jsonify({"prediction": result})
    except:
        return jsonify({"error": "Invalid Input"}), 400


###############################################################################
# CLUSTERING PAGE: User picks k, we do KMeans, show a 2D scatter
###############################################################################
@app.route('/clustering')
def clustering_page():
    """
    Single-page UI for K-Means clustering with user input for number of clusters (k).
    """
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Iris Clustering</title>
        <link rel="stylesheet"
              href="https://cdn.jsdelivr.net/npm/bootstrap@4.6.2/dist/css/bootstrap.min.css">
        <style>
            body {
                background: #f5f5f7;
                color: #1d1d1f;
                font-family: -apple-system, BlinkMacSystemFont,
                             'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            }
            .navbar { background-color: #000; }
            .navbar-brand, .nav-link { color: #fff !important; }
            .hero { text-align: center; padding: 30px 20px; }
            .hero h1 { font-size: 2rem; font-weight: 600; margin-bottom: 10px; }
            .cluster-form { max-width: 400px; margin: 0 auto; margin-top: 20px; }
            .btn-apple {
                background-color: #0070c9;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 1rem;
                border-radius: 50px;
                transition: background-color 0.3s ease, transform 0.2s;
            }
            .btn-apple:hover {
                background-color: #005ea6;
                transform: translateY(-2px);
            }
            #cluster-image {
                max-width: 600px;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <nav class="navbar navbar-expand-lg">
            <a class="navbar-brand" href="/">Iris ML Experiments</a>
        </nav>

        <div class="hero">
            <h1>K-Means Clustering</h1>
        </div>

        <div class="container text-center">
            <p>Enter the number of clusters (k) and click "Show Clusters" to visualize.</p>
            
            <form class="cluster-form">
              <div class="form-group">
                <input type="number" min="2" max="10" class="form-control" id="inputK"
                       placeholder="Number of Clusters (k)" required />
              </div>
              <button type="submit" class="btn-apple">Show Clusters</button>
            </form>

            <img id="cluster-image" src="" alt="Cluster Plot" />

            <br><br>
            <a href="/" class="btn-apple">Go Back Home</a>
        </div>

        <script>
        const form = document.querySelector('.cluster-form');
        const imgEl = document.getElementById('cluster-image');

        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            const kVal = document.getElementById('inputK').value;

            if (!kVal) {
                alert("Please enter a valid k.");
                return;
            }

            try {
                const response = await fetch('/plot_clusters', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ k: kVal })
                });
                
                if(!response.ok) {
                    throw new Error("Network response was not OK");
                }
                const data = await response.json();
                // data.plot_url is a base64 data URL
                imgEl.src = data.plot_url;
            } catch(err) {
                console.error("Error:", err);
                alert("Failed to get cluster plot.");
            }
        });
        </script>
    </body>
    </html>
    """


@app.route('/plot_clusters', methods=['POST'])
def plot_clusters():
    """
    Endpoint: Takes a JSON with 'k'. Runs K-Means (k clusters) on Iris data,
    plots a 2D scatter with cluster labels, returns base64 image data.
    """
    try:
        data = request.get_json(force=True)
        k = int(data['k'])
        if k < 2 or k > 10:
            return jsonify({"error": "k must be between 2 and 10"}), 400

        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_iris)
        labels = kmeans.labels_

        # We'll plot using the first two features: sepal_length, sepal_width
        x_ = X_iris[:, 0]
        y_ = X_iris[:, 1]

        fig, ax = plt.subplots(figsize=(6, 4))
        scatter = ax.scatter(x_, y_, c=labels, cmap='viridis', s=40)
        ax.set_xlabel("Sepal Length")
        ax.set_ylabel("Sepal Width")
        ax.set_title(f"K-Means Clusters (k={k})")

        # Convert plot to base64
        pngImage = io.BytesIO()
        plt.savefig(pngImage, format='png', bbox_inches='tight')
        plt.close(fig)
        pngImage.seek(0)

        base64Image = base64.b64encode(pngImage.read()).decode('utf-8')
        plot_url = "data:image/png;base64," + base64Image
        return jsonify({"plot_url": plot_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    """
    When running locally:
    python app.py
    -> open http://127.0.0.1:5000/
    """
    app.run(host='0.0.0.0', port=5000)
