from sklearn.base import DensityMixin


WIDTH           =    640
HEIGHT          =    480
WIDTH_SLICE     =    50
HEIGHT_SLICE    =    20
LOW_CUT_DEPTH   =    0.002             
HIGH_CUT_DEPTH  =    0.1                 
INTERVAL        =    0.13                # camera interval
BOX_WIDTH       =    1.8                 # width of rectangle box 
BOX_HEIGHT      =    0.6                 # height of rectangle box
DENSITY         =    1000                # 1 meter / Density = sampling unit
CAMERA_MODE     =    'v'                 # 'h': horizontal mode, 'v': vertical mode 
EPS             =    10                # eps of DBSCAN
MIN_SAMPLES     =    8                 # min samples of DBSCAN 

# vertical 모드에서는, 카메라 기준 아래 방향으로 이동을 시켜야합니다!!