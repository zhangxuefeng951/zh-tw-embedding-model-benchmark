from mock_server import fake_embedding


def fake_vec(text: str):
    return fake_embedding(text, 64)
