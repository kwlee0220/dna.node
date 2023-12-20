
from .object_detector import ObjectDetector

def load_object_detector(uri: str) -> ObjectDetector:
    if not uri:
        raise ValueError(f"detector id is None")

    parts = uri.split(':', 1)
    id, query = tuple(parts) if len(parts) > 1 else (uri, "")
    if id == 'file':
        from pathlib import Path
        from .object_detector import LogReadingDetector

        det_file = Path(query)
        return LogReadingDetector(det_file)
    else:
        import importlib
        
        loader_module = importlib.import_module(id)
        return loader_module.load(query)