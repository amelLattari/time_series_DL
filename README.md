# Time Series Deep Learning Project  
## Étudiantes :  

    Amel Lattari  
    Amina Tadjin  
    Nour Ikhelef  
  
## Description du Projet  

  * Ce projet vise à construire et déployer un modèle de prévision basé sur des séries temporelles en utilisant une architecture combinée CNN-LSTM. Le pipeline couvre toutes les étapes, de la construction du modèle à son déploiement avec des outils MLOps.

## Structure du Projet  
  * Fichiers Racine
    - app.py : Script principal contenant l'API Flask pour servir le modèle.  
    - Dockerfile : Fichier pour containeriser l'application avec Docker.  
    - alert_rules.yml : Règles de monitoring pour Prometheus.  
    - deployment.yaml : Fichier de configuration Kubernetes pour déployer l'API et le service.  
    - docker-compose.yml : Configuration pour orchestrer la stack ELK (Elasticsearch, Logstash, Kibana).  
    - logstash.conf : Configuration de Logstash pour centraliser les logs.  
    - prometheus.yml : Configuration de Prometheus pour surveiller les métriques.  
    - requirments.txt : Liste des dépendances nécessaires au projet.  
    - sales_forecasting_model_v3.h5 : Modèle de prévision des ventes sauvegardé après entraînement.
    - Projet_Ml_M2.ipynb : Notebook couvrant l'analyse des séries temporelles avec les modèles ARIMA et SARIMA, ainsi que la création, l'entraînement et le déploiement du modèle CNN-LSTM.
  * Dossiers  
    - templates/ : Contient les fichiers HTML.  
    - index.html : Interface utilisateur pour interagir avec l'API.  
    - static/ : Contient les fichiers CSS.  
    - style.css : Feuille de style pour l'interface utilisateur.  
