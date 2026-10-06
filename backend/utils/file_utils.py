from pathlib import Path
from werkzeug.utils import secure_filename


def ensure_directory(directory):
    """
    Create a directory if it does not already exist.
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def is_allowed_file(filename, allowed_extensions):
    """
    Check whether a file has an allowed extension.
    """
    if not filename:
        return False

    extension = Path(filename).suffix.lower()

    return extension in allowed_extensions


def secure_file_name(filename):
    """
    Make the uploaded filename safe.
    """
    return secure_filename(filename)