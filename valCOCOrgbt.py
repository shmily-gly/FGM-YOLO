# # 这是专门的COCO指标计算方法，请参考我的使用方法
# # 我的这些参数方法是可以调整，请你根据你的项目调试    PYTHONPATH=/home/gly/code/yolomm python -u valCOCOrgbt.py >> test/yolo11n-mm-mid.txt 2>&1
# from ultralytics import YOLOMM
# import os
# # os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# model = YOLOMM('/home/gly/code/yolomm/runs/rgbt/yolo11n-mm-mid/weights/best.pt')
# model.cocoval(data='/home/gly/code/yolomm/data.yaml',
#               batch=16, 
#               imgsz=640,
#               device=[0],
#               seed=0,
#               exist_ok=False,
#               workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#               split='val',
#               half=False,
#               project='valcoco',
#               name='yolo11n-mm-mid')






# # 这是专门的COCO指标计算方法，请参考我的使用方法
# # 我的这些参数方法是可以调整，请你根据你的项目调试    PYTHONPATH=/home/gly/code/yolomm python -u valCOCOrgbt.py >> test/yolo11n-mm-mid-p2.txt 2>&1
# from ultralytics import YOLOMM
# import os
# # os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# model = YOLOMM('/home/gly/code/yolomm/runs/rgbt/yolo11n-mm-mid-p2/weights/best.pt')
# model.cocoval(data='/home/gly/code/yolomm/data.yaml',
#               batch=16, 
#               imgsz=640,
#               device=[0],
#               seed=0,
#               exist_ok=False,
#               workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#               split='val',
#               half=False,
#               project='valcoco',
#               name='yolo11n-mm-mid-p2')



# # 这是专门的COCO指标计算方法，请参考我的使用方法
# # 我的这些参数方法是可以调整，请你根据你的项目调试    PYTHONPATH=/home/gly/code/yolomm python -u valCOCOrgbt.py >> test/yolo11n-mm-mid-p2rgb-logstem.txt 2>&1
# from ultralytics import YOLOMM
# import os
# # os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# model = YOLOMM('/home/gly/code/yolomm/runs/rgbt/yolo11n-mm-mid-p2rgb-logstem/weights/best.pt')
# model.cocoval(data='/home/gly/code/yolomm/data.yaml',
#               batch=16, 
#               imgsz=640,
#               device=[0],
#               seed=0,
#               exist_ok=False,
#               workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#               split='val',
#               half=False,
#               project='valcoco',
#               name='yolo11n-mm-mid-p2rgb-logstem')



# 这是专门的COCO指标计算方法，请参考我的使用方法
# # 我的这些参数方法是可以调整，请你根据你的项目调试    PYTHONPATH=/home/gly/code/yolomm python -u valCOCOrgbt.py >> test/yolo11n-mm-mid-soep.txt 2>&1
# from ultralytics import YOLOMM
# import os
# # os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# model = YOLOMM('/home/gly/code/yolomm/runs/rgbt/yolo11n-mm-mid-soep/weights/best.pt')
# model.cocoval(data='/home/gly/code/yolomm/data.yaml',
#               batch=16, 
#               imgsz=640,
#               device=[0],
#               seed=0,
#               exist_ok=False,
#               workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#               split='val',
#               half=False,
#               project='valcoco',
#               name='yolo11n-mm-mid-soep')




# # 这是专门的COCO指标计算方法，请参考我的使用方法
# # 我的这些参数方法是可以调整，请你根据你的项目调试    PYTHONPATH=/home/gly/code/yolomm python -u valCOCOrgbt.py >> test/yolo11n-mm-mid-p2rgb-dysample.txt 2>&1
# from ultralytics import YOLOMM
# import os
# # os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# model = YOLOMM('/home/gly/code/yolomm/runs/rgbt/yolo11n-mm-mid-p2rgb-dysample/weights/best.pt')
# model.cocoval(data='/home/gly/code/yolomm/data.yaml',
#               batch=16, 
#               imgsz=640,
#               device=[0],
#               seed=0,
#               exist_ok=False,
#               workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#               split='val',
#               half=False,
#               project='valcoco',
#               name='yolo11n-mm-mid-p2rgb-dysample')




# # 这是专门的COCO指标计算方法，请参考我的使用方法
# # 我的这些参数方法是可以调整，请你根据你的项目调试    PYTHONPATH=/home/gly/code/yolomm python -u valCOCOrgbt.py >> test/yolo11n-mm-mid-p2-ir.txt 2>&1
# from ultralytics import YOLOMM
# import os
# # os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# model = YOLOMM('/home/gly/code/yolomm/runs/rgbt/yolo11n-mm-mid-p2-ir/weights/best.pt')
# model.cocoval(data='/home/gly/code/yolomm/data.yaml',
#               batch=16, 
#               imgsz=640,
#               device=[0],
#               seed=0,
#               exist_ok=False,
#               workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#               split='val',
#               half=False,
#               project='valcoco',
#               name='yolo11n-mm-mid-p2-ir')




# # 这是专门的COCO指标计算方法，请参考我的使用方法
# # 我的这些参数方法是可以调整，请你根据你的项目调试    PYTHONPATH=/home/gly/code/yolomm python -u valCOCOrgbt.py >> test/yolo11n-mm-mid-p2-rgbir.txt 2>&1
# from ultralytics import YOLOMM
# import os
# # os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# model = YOLOMM('/home/gly/code/yolomm/runs/rgbt/yolo11n-mm-mid-p2-rgbir/weights/best.pt')
# model.cocoval(data='/home/gly/code/yolomm/data.yaml',
#               batch=16, 
#               imgsz=640,
#               device=[0],
#               seed=0,
#               exist_ok=False,
#               workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#               split='val',
#               half=False,
#               project='valcoco',
#               name='yolo11n-mm-mid-p2-rgbir')




# # 这是专门的COCO指标计算方法，请参考我的使用方法
# # 我的这些参数方法是可以调整，请你根据你的项目调试    PYTHONPATH=/home/gly/code/yolomm python -u valCOCOrgbt.py >> test/yolo11n-mm-mid-p2rgb-p3ir.txt 2>&1
# from ultralytics import YOLOMM
# import os
# # os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# model = YOLOMM('/home/gly/code/yolomm/runs/rgbt/yolo11n-mm-mid-p2rgb-p3ir/weights/best.pt')
# model.cocoval(data='/home/gly/code/yolomm/data.yaml',
#               batch=16, 
#               imgsz=640,
#               device=[0],
#               seed=0,
#               exist_ok=False,
#               workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#               split='val',
#               half=False,
#               project='valcoco',
#               name='yolo11n-mm-mid-p2rgb-p3ir')




# # 这是专门的COCO指标计算方法，请参考我的使用方法
# # 我的这些参数方法是可以调整，请你根据你的项目调试    PYTHONPATH=/home/gly/code/yolomm python -u valCOCOrgbt.py >> test/yolo11n-mm-mid-p2rgb-p3rgbir.txt 2>&1
# from ultralytics import YOLOMM
# import os
# # os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# model = YOLOMM('/home/gly/code/yolomm/runs/rgbt/yolo11n-mm-mid-p2rgb-p3rgbir/weights/best.pt')
# model.cocoval(data='/home/gly/code/yolomm/data.yaml',
#               batch=16, 
#               imgsz=640,
#               device=[0],
#               seed=0,
#               exist_ok=False,
#               workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#               split='val',
#               half=False,
#               project='valcoco',
#               name='yolo11n-mm-mid-p2rgb-p3rgbir')





# # 这是专门的COCO指标计算方法，请参考我的使用方法
# # 我的这些参数方法是可以调整，请你根据你的项目调试    PYTHONPATH=/home/gly/code/yolomm python -u valCOCOrgbt.py >> test/yolo11n-mm-mid-p2rgb-p45-mcf.txt 2>&1
# from ultralytics import YOLOMM
# import os
# # os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# model = YOLOMM('/home/gly/code/yolomm/runs/rgbt/yolo11n-mm-mid-p2rgb-p45-mcf/weights/best.pt')
# model.cocoval(data='/home/gly/code/yolomm/data.yaml',
#               batch=16, 
#               imgsz=640,
#               device=[0],
#               seed=0,
#               exist_ok=False,
#               workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#               split='val',
#               half=False,
#               project='valcoco',
#               name='yolo11n-mm-mid-p2rgb-p45-mcf')




# # 这是专门的COCO指标计算方法，请参考我的使用方法
# # 我的这些参数方法是可以调整，请你根据你的项目调试    PYTHONPATH=/home/gly/code/yolomm python -u valCOCOrgbt.py >> test/yolo11n-mm-mid-p2rgb-p45-mfm.txt 2>&1
# from ultralytics import YOLOMM
# import os
# # os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# model = YOLOMM('/home/gly/code/yolomm/runs/rgbt/yolo11n-mm-mid-p2rgb-p45-mfm/weights/best.pt')
# model.cocoval(data='/home/gly/code/yolomm/data.yaml',
#               batch=16, 
#               imgsz=640,
#               device=[0],
#               seed=0,
#               exist_ok=False,
#               workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#               split='val',
#               half=False,
#               project='valcoco',
#               name='yolo11n-mm-mid-p2rgb-p45-mfm')




# # # 这是专门的COCO指标计算方法，请参考我的使用方法
# # 我的这些参数方法是可以调整，请你根据你的项目调试    PYTHONPATH=/home/gly/code/yolomm python -u valCOCOrgbt.py >> test/yolo11n-mm-mid-soep-rfpn.txt 2>&1
# from ultralytics import YOLOMM
# import os
# # os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# model = YOLOMM('/home/gly/code/yolomm/runs/rgbt/yolo11n-mm-mid-soep-rfpn/weights/best.pt')
# model.cocoval(data='/home/gly/code/yolomm/data.yaml',
#               batch=16, 
#               imgsz=640,
#               device=[0],
#               seed=0,
#               exist_ok=False,
#               workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#               split='val',
#               half=False,
#               project='valcoco',
#               name='yolo11n-mm-mid-soep-rfpn')





# # # 这是专门的COCO指标计算方法，请参考我的使用方法
# # 我的这些参数方法是可以调整，请你根据你的项目调试    PYTHONPATH=/home/gly/code/yolomm python -u valCOCOrgbt.py >> test/yolo11n-mm-mid-p2rgb-logstem-dysample-3.txt 2>&1
# from ultralytics import YOLOMM
# import os
# # os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# model = YOLOMM('/home/gly/code/yolomm/runs/rgbt/yolo11n-mm-mid-p2rgb-logstem-dysample-3/weights/best.pt')
# model.cocoval(data='/home/gly/code/yolomm/data.yaml',
#               batch=16, 
#               imgsz=640,
#               device=[0],
#               seed=0,
#               exist_ok=False,
#               workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#               split='val',
#               half=False,
#               project='valcoco',
#               name='yolo11n-mm-mid-p2rgb-logstem-dysample-3')




# # # 这是专门的COCO指标计算方法，请参考我的使用方法
# # 我的这些参数方法是可以调整，请你根据你的项目调试    PYTHONPATH=/home/gly/code/yolomm python -u valCOCOrgbt.py >> test/yolo11n-mm-mid-p2rgb-logstem-p3rgbir.txt 2>&1
# from ultralytics import YOLOMM
# import os
# # os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# model = YOLOMM('/home/gly/code/yolomm/runs/rgbt/yolo11n-mm-mid-p2rgb-logstem-p3rgbir/weights/best.pt')
# model.cocoval(data='/home/gly/code/yolomm/data.yaml',
#               batch=16, 
#               imgsz=640,
#               device=[0],
#               seed=0,
#               exist_ok=False,
#               workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#               split='val',
#               half=False,
#               project='valcoco',
#               name='yolo11n-mm-mid-p2rgb-logstem-p3rgbir')




# # 这是专门的COCO指标计算方法，请参考我的使用方法
# 我的这些参数方法是可以调整，请你根据你的项目调试    PYTHONPATH=/home/gly/code/yolomm python -u valCOCOrgbt.py >> test/yolo11n-mm-mid-p2rgb-logstem-p3mjrnet.txt 2>&1
from ultralytics import YOLOMM
import os
# os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

model = YOLOMM('/home/gly/code/yolomm/runs/rgbt/yolo11n-mm-mid-p2rgb-logstem-p3mjrnet/weights/best.pt')
model.cocoval(data='/home/gly/code/yolomm/data.yaml',
              batch=16, 
              imgsz=640,
              device=[0],
              seed=0,
              exist_ok=False,
              workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
              split='val',
              half=False,
              project='valcoco',
              name='yolo11n-mm-mid-p2rgb-logstem-p3mjrnet')
            