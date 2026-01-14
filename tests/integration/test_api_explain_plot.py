from PIL import Image
import io
import pytest

def explain_plot_ok(client, valid_record):
    response = client.post("/explain/plot", json=valid_record)
    assert response.status_code == 200

    assert response.headers["media_type"] == "image/png"
    assert isinstance(response.content, bytes)

    try:
        img = Image.open(io.BytesIO(response.content))
        assert img.format == "PNG"
        img.verify()
    except Exception as e:
        pytest.fail(f"Response content is not a valid image: {e}")



