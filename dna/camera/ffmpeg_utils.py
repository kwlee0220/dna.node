from __future__ import annotations

from typing import Optional
import os
from pathlib import Path
from subprocess import Popen, DEVNULL

_ffmpeg_path:str = None


def is_initialize() -> bool:
    return _ffmpeg_path is not None


def init(*, ffmpeg_path:Optional[str|Path]=None) -> None:
    global _ffmpeg_path
    
    if _ffmpeg_path is not None:
        if _ffmpeg_path.samefile(Path(ffmpeg_path)):
            return
    
    if ffmpeg_path is None:
        ffmpeg_path = os.environ.get('FFMPEG_PATH')
        if ffmpeg_path is None:
            raise ValueError(f"ffmepg execution file is not specified: check the environment variable 'FFMPEG_PATH'")
            
    ffmpeg_path = Path(ffmpeg_path)
    if not (ffmpeg_path.exists() and ffmpeg_path.is_file()):
        raise ValueError(f"FFMPEG execution file is not found: path={ffmpeg_path}")
    
    _ffmpeg_path = str(ffmpeg_path)
        

def write_rtsp_stream(rstp_url:str, video_file:str,
                      *,
                      timeout:Optional[float]=None) -> int:
    if _ffmpeg_path is None:
        raise ValueError(f"ffmpeg module is not initialized")
    
    ffmpeg_cmd = [_ffmpeg_path, "-rtsp_transport", "tcp", "-i", rstp_url, "-c:v", "copy", "-f", "mp4", "-y", video_file]        
    proc = Popen(ffmpeg_cmd, stdout=DEVNULL, stderr=DEVNULL)
    return proc.wait(timeout)
    

def relay_to_local_stream(input_rstp_url:str, id:str) -> tuple[Popen, str]:
    if _ffmpeg_path is None:
        raise ValueError(f"ffmpeg module is not initialized")
    
    output_rtsp_url = f"rtsp://localhost:8554/{id}"
    ffmpeg_cmd = [_ffmpeg_path, "-rtsp_transport", "tcp", "-i", input_rstp_url,
                  "-rtsp_transport", "tcp", "-c:v", "copy", "-f", "rtsp", output_rtsp_url]
    return Popen(ffmpeg_cmd, stdout=DEVNULL, stderr=DEVNULL), output_rtsp_url