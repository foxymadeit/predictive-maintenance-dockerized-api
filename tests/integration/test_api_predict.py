def test_predict_ok(client, valid_record):
    response = client.post("/predict", json=valid_record)
    assert response.status_code == 200

    data = response.json()
    assert "proba_failure" in data
    assert "alert" in data
    assert "threshold" in data

    assert 0.0 <= data["threshold"] <= 1.0
    assert data["alert"] in (0, 1)


def test_predict_error(client, invalid_record):
    response = client.post('/predict', json=invalid_record)
    assert response.status_code == 422