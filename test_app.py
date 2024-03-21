import pytest
from app import app as flask_app  # Importez votre application Flask

@pytest.fixture
def app():
    yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()

def test_predict_endpoint(client):
    # Préparez les données de test et les headers
    test_data = {'text': 'Exemple de texte pour la prédiction.'}
    headers = {'Content-Type': 'application/json'}
    
    # Effectuez une requête POST vers l'endpoint /predict
    response = client.post('/predict', json=test_data, headers=headers)
    
    # Vérifiez que la requête a réussi
    assert response.status_code == 200
    
    # Vérifiez la structure de la réponse
    json_data = response.get_json()
    assert 'prediction' in json_data