# 这是专门的COCO指标计算方法，请参考我的使用方法
# 我的这些参数方法是可以调整，请你根据你的项目调试    PYTHONPATH=/home/gly/code/yolomm python -u valCOCO.py >> test/yolo11n-mm-mid.txt 2>&1
from ultralytics import YOLOMM
import os
# os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

model = YOLOMM('/home/gly/code/yolomm/runs/yolo11n-mm-mid/weights/best.pt')
model.cocoval(data='/home/gly/code/yolomm/data.yaml',
              batch=16, 
              imgsz=640,
              device=[0],
            #   device=[0,1],
              seed=0,
              exist_ok=False,
              workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
              split='val',
              half=False,
              project='valcoco',
              name='yolo11n-mm-mid')




# # 这是专门的COCO指标计算方法，请参考我的使用方法
# # 我的这些参数方法是可以调整，请你根据你的项目调试    PYTHONPATH=/home/gly/code/yolomm python -u valCOCO.py >> test/yolo11n-mm-mid-p2.txt 2>&1
# from ultralytics import YOLOMM
# import os
# # os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# model = YOLOMM('/home/gly/code/yolomm/runs/yolo11n-mm-mid-p2/weights/best.pt')
# model.cocoval(data='/home/gly/code/yolomm/data.yaml',
#               batch=16, 
#               imgsz=640,
#               device=[0],
#             #   device=[0,1],
#               seed=0,
#               exist_ok=False,
#               workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#               split='val',
#               half=False,
#               project='valcoco',
#               name='yolo11n-mm-mid-p2')






# # 这是专门的COCO指标计算方法，请参考我的使用方法
# # 我的这些参数方法是可以调整，请你根据你的项目调试    PYTHONPATH=/home/gly/code/yolomm python -u valCOCO.py >> test/yolo11n-mm-mid-soep.txt 2>&1
# from ultralytics import YOLOMM
# import os
# # os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# model = YOLOMM('/home/gly/code/yolomm/runs/yolo11n-mm-mid-soep/weights/best.pt')
# model.cocoval(data='/home/gly/code/yolomm/data.yaml',
#               batch=16, 
#               imgsz=640,
#               device=[0],
#             #   device=[0,1],
#               seed=0,
#               exist_ok=False,
#               workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#               split='val',
#               half=False,
#               project='valcoco',
#               name='yolo11n-mm-mid-soep')







# # 这是专门的COCO指标计算方法，请参考我的使用方法
# # 我的这些参数方法是可以调整，请你根据你的项目调试    PYTHONPATH=/home/gly/code/yolomm python -u valCOCO.py >> test/yolo11n-mm-mid-soep-rfpn.txt 2>&1
# from ultralytics import YOLOMM
# import os
# # os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# model = YOLOMM('/home/gly/code/yolomm/runs/yolo11n-mm-mid-soep-rfpn/weights/best.pt')
# model.cocoval(data='/home/gly/code/yolomm/data.yaml',
#               batch=16, 
#               imgsz=640,
#               device=[0],
#             #   device=[0,1],
#               seed=0,
#               exist_ok=False,
#               workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#               split='val',
#               half=False,
#               project='valcoco',
#               name='yolo11n-mm-mid-soep-rfpn')