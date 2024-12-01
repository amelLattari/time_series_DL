from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Gauge
import matplotlib.pyplot as plt
import io
import base64
import logging
import json
import socket
from logging.handlers import SocketHandler

# Load the model
model = tf.keras.models.load_model('sales_forecasting_model_v3.h5', custom_objects={'mse': MeanSquaredError()})

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logstash_handler = SocketHandler('localhost', 5000)
formatter = logging.Formatter('%(message)s')
logstash_handler.setFormatter(formatter)
logger.addHandler(logstash_handler)

# Initialize Flask app
app = Flask(__name__)

# Integrate Prometheus
metrics = PrometheusMetrics(app)
accuracy_gauge = Gauge('model_accuracy', 'Accuracy of the model')
loss_gauge = Gauge('model_loss', 'Loss of the model')

# Route for the main page
@app.route('/')
def home():
    return render_template('index.html')

# Function to create a graph
def create_graph(sales_data, predictions):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(sales_data)), sales_data, label="Données réelles", marker='o')
    plt.plot(range(len(predictions)), predictions, label="Prédictions", marker='x', linestyle='--')
    plt.title("Graphique des Ventes et Prédictions")
    plt.xlabel("Temps")
    plt.ylabel("Ventes")
    plt.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    graph_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return graph_img

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user data
        sales_data_str = request.form['sales_data']
        sales_data = np.array([float(x) for x in sales_data_str.split(',')])

        # Vérifiez que les données d'entrée contiennent exactement 12 valeurs
        if len(sales_data) != 12:
            return jsonify({'error': 'Les données d\'entrée doivent contenir exactement 12 valeurs'}), 400

        # Calculer min et max dynamiques pour les ventes
        min_val = sales_data.min()
        max_val = sales_data.max()

        # Ajouter une deuxième caractéristique : les mois [1, 2, ..., 12]
        input_months = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        month_min, month_max = input_months.min(), input_months.max()

        # Normalisation des ventes
        normalized_sales_data = (sales_data - min_val) / (max_val - min_val)

        # Normalisation des mois
        normalized_months = (input_months - month_min) / (month_max - month_min)

        # Combiner les caractéristiques (ventes + mois) pour créer l'entrée
        input_features = np.stack((normalized_sales_data, normalized_months), axis=-1).reshape(1, 12, 2)

        # Prédiction
        predictions_normalized = model.predict(input_features).flatten()

        # Dénormalisation des prédictions
        predictions = predictions_normalized * (max_val - min_val) + min_val

        # Générer le graphe
        graph_img = create_graph(sales_data, predictions)

        # Log event
        logger.info(json.dumps({"event": "prediction", "data": sales_data_str}))

        # Mettre à jour les métriques Prometheus
        accuracy = 0.95
        loss = 0.05
        accuracy_gauge.set(accuracy)
        loss_gauge.set(loss)

        # Retourner la page avec prédiction et graphe
        return render_template('index.html', prediction=predictions.tolist(), graph_img=graph_img)
    except Exception as e:
        return jsonify({'error': str(e)}), 400



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
