import os

def get_imagej_obj(fiji_path=None):
    import imagej
    from scyjava import jimport
    if not fiji_path:
        fiji_path = '/home/ryan/Documents/Fiji.app'

    if os.path.isdir(fiji_path):
        ij = imagej.init(fiji_path, headless=False)
        return ij
    else:
        print("Cannot proceed: Fiji not found!")


