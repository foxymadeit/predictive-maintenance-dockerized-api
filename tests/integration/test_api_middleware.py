import re
import logging

def test_middleware_ok(client):
    response = client.get(f"{client.base_url}")
    assert response.status_code == 200


    request_id = response.headers.get("X-Request-ID")
    uuid_regex = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"

    assert len(response.headers["X-Request-ID"]) == 36
    assert re.match(uuid_regex, request_id.lower())
    assert isinstance(response.headers["X-Request-ID"], str)



def test_middleware_uses_provided_request_id(client):
    external_id = "test-external-id-123"
    response = client.get(f"{client.base_url}", headers={"X-Request-ID": external_id})
    
    assert response.headers["X-Request-ID"] == external_id



def test_middleware_log(client, caplog):
    response = client.get(f"{client.base_url}")
    for record in caplog.records:
        assert record.levelname == "INFO"
        assert "timestamp" in record
        assert "request completed" in record
        assert "latency_ms" > 0




def test_middleware_id_on_server_error(client, caplog):
    logging.getLogger("api").propagate = True
    
    with caplog.at_level(logging.INFO, logger="api"):
        client.get("/invalid_endpoint")
        
    log_record = caplog.records[-1]
    
    assert hasattr(log_record, "extra")
    
    log_data = log_record.extra
    assert log_data["request_id"] is not None
    assert log_data["status_code"] == 404
    assert log_data["method"] == "GET"


    