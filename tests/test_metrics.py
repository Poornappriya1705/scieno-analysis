from src.scientometric.metrics import h_index

def test_h_index(): assert h_index([10,8,5,4,3])==4
