def test_predict_batch_ok(client, valid_record):
    batch_load = {
        "records": [valid_record, valid_record]
    }

    response = client.post("/predict/batch", json=batch_load)
    assert response.status_code == 200

    data = response.json()
    
    assert isinstance(data, dict)
    assert "proba_failure" in data["results"][0]