from sklearn.base import DensityMixin


WIDTH           =    640
HEIGHT          =    480             
INTERVAL        =    0.25                # camera interval
BOX_WIDTH       =    1.2                 # width of rectangle box 
BOX_HEIGHT      =    0.6                 # height of rectangle box
DENSITY         =    1000                # 1 meter / Density = sampling unit
CAMERA_MODE     =    'v'                 # 'h': horizontal mode, 'v': vertical mode 

# vertical 모드에서는, 카메라 기준 아래 방향으로 이동을 시켜야합니다!!