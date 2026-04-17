import zipfile
import os
import io

def create_zip(file_paths):
    """
    Accepts a list of file paths and returns a BytesIO zip buffer.
    """
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for path in file_paths:
            if os.path.exists(path):
                zipf.write(path, arcname=os.path.basename(path))

    buffer.seek(0)
    return buffer


'''import zipfile
import os
from io import BytesIO

def create_zip(file_paths):
    zip_buffer = BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in file_paths:
            zf.write(path, arcname=os.path.basename(path))

    zip_buffer.seek(0)
    return zip_buffer.getvalue()'''




'''import io
import zipfile


def create_zip(files_dict):
    """
    files_dict = {filename: bytes}
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for fname, fbytes in files_dict.items():
            zipf.writestr(fname, fbytes)
    buffer.seek(0)
    return buffer'''
