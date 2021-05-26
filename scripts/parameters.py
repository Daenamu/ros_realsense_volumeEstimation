WIDTH           =    640
HEIGHT          =    480
WIDTH_SLICE     =    50
HEIGHT_SLICE    =    20
LOW_CUT_DEPTH   =    0.015               # low cut under 1.5cm error   
HIGH_CUT_DEPTH  =    0.1                 # high cut over 10cm error
INTERVAL        =    0.15                # camera interval is 15cm
CAMERA_MODE     =    'v'                 # 'h': horizontal mode, 'v': vertical mode 
EPS             =    10                # eps of DBSCAN
MIN_SAMPLES     =    8                 # min samples of DBSCAN 

# vertical 모드에서는, 카메라 기준 아래 방향으로 이동을 시켜야합니다!!