import autoballs.ijconn as ijconn

def test_get_imagej_obj():
    ij = ijconn.get_imagej_obj()
    assert ij != None
