import os

def test_PYTHONPATH():
    # set XX to path of root directory
    print(os.environ.get('PYTHONPATH') == "XX")
    assert(os.environ.get('PYTHONPATH') == "XX")
