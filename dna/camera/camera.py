from __future__ import annotations

from typing import Any

from .types import Camera, CameraOptions


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


def load_camera(camera_uri:str, **options) -> Camera:
    """Create an OpenCvCamera object of the given URI.
    The additional options will be given by dictionary ``options``.
    The options contain the followings:
    - size: the size of the image that the created camera will capture (optional)

    Args:
        camera_uri (str): id of the camera.
        
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
    
    cam_opts = CameraOptions(**options)
    
    if is_video_file(camera_uri):
        return VideoFile(camera_uri, cam_opts)
    elif is_local_camera(camera_uri):
        return OpenCvCamera(camera_uri, cam_opts)
    elif is_rtsp_camera(camera_uri):
        if camera_uri.find("&end=") >= 0 or camera_uri.find("start=") >= 0:
            from .ffmpeg_camera import FFMPEGCamera
            return FFMPEGCamera(camera_uri, cam_opts)
        else:
            return OpenCvCamera(camera_uri, cam_opts)
    else:
        raise ValueError(f'invalid Camera URI: {camera_uri}')
    

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