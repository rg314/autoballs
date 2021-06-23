import os

def get_imagej_obj(fiji_path=None, headless=False):
    import imagej
    from scyjava import jimport
    print(os.getcwd())
    if not fiji_path:
        fiji_path = os.getcwd() + '/Fiji/Fiji.app'

    if os.path.isdir(fiji_path):
        ij = imagej.init(fiji_path, headless=headless)
        return ij
    else:
        msg = "Cannot proceed: Fiji not found! Please run 'get_fiji_version.sh'"
        raise OSError()



