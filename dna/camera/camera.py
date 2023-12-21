from __future__ import annotations

from typing import Any, Optional

from omegaconf import OmegaConf

from .types import Frame, Camera
from .opencv_camera import OpenCvCamera, VideoFile


def is_local_camera(uri:str):
    '''Determines that the give URI is for the local camera or not.
    'Local camera' means the one is directly connected to this computer through USB or any other media.'''
    return uri.isnumeric()


def is_video_file(uri:str):
    '''Determines whether the images captured from a video file or not.'''
    return uri.endswith('.mp4') or uri.endswith('.avi')


def is_rtsp_camera(uri:str):
    '''Determines whether the camera of the give URI is a remote one accessed by the RTSP protocol.'''
    return uri.startswith('rtsp://')


def create_camera_from_conf(conf:OmegaConf|dict[str,Any]) -> Camera:
    """Create a camera from OmegaConf configuration.

    Args:
        conf (OmegaConf|dict[str,Any]): configuration.

    Returns:
        OpenCvCamera: an OpenCvCamera object.
    """
    if isinstance(conf, OmegaConf):
        conf = dict(conf)
    options = {k:v for k, v in conf.items() if k != 'uri'}
    return load_camera(conf.uri, **options)
    

def load_camera(uri:str, **options) -> Camera:
    """Create an OpenCvCamera object of the given URI.
    The additional options will be given by dictionary ``options``.
    The options contain the followings:
    - size: the size of the image that the created camera will capture (optional)

    Args:
        uri (str): id of the camera.
        
    Keyward Args:
        size (str): image size
        init_ts (str): initial timestamp for the first frame.
        sync (bool): synchronized image capturing or not
        begin_frame (int): the first frame to capture.
        end_frame (int): the last frame to capture.

    Returns:
        OpenCvCamera: an OpenCvCamera object.
        If URI points to a video file, ``OpenCvVideFile`` object is returned. Otherwise,
        ``OpenCvCamera`` is returned.
    """
    from .opencv_camera import OpenCvCamera, VideoFile
    
    if is_video_file(uri):
        return VideoFile(uri, **options)
    elif is_local_camera(uri):
        return OpenCvCamera(int(uri), **options)
    elif is_rtsp_camera(uri):
        if uri.find("&end=") >= 0 or uri.find("start=") >= 0:
            from .ffmpeg_camera import FFMPEGCamera
            return FFMPEGCamera(uri, **options)
        else:
            return OpenCvCamera(uri, **options)
    else:
        raise ValueError(f'invalid Camera URI: {uri}')
    

# import reactivex as rx    
# from reactivex import Observer, Observable
# def observe(camera:Camera) -> Observable[Frame]:
#     def supply_frames(observer:Observer[Frame], scheduler):
#         with camera.open() as cap:
#             try:
#                 for frame in cap:
#                     observer.on_next(frame)
#                 observer.on_completed()
#             except Exception as error:
#                 observer.on_error(error)
#     return rx.create(supply_frames)