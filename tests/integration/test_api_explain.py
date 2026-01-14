def test_explain_ok(client, valid_record):
    response = client.post("/explain", json=valid_record)
    assert response.status_code == 200

    data = response.json()

    assert "proba_failure" in data
    assert "alert" in data
    assert "threshold" in data
    assert "top_contributors" in data

    assert data["alert"] in (0,1)
    assert 0.0 <= data["threshold"] <= 1.0
    assert isinstance(data["proba_failure"], float)