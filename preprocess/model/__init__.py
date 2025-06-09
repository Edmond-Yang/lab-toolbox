import os

method = os.environ.get('METHOD', 'mediapipe')

if method == 'mtcnn':
    from .MTCNN import *
elif method == 'mediapipe':
    from .GoogleMediaPipe import *
else:
    from .RetinaFaceDetect import *
