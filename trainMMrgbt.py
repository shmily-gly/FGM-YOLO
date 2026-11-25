# #这是提供 参考示例的 YOLOMM训练器使用方法
# #按必须按照你的需求来更改而不是盲目去使用   PYTHONPATH=/home/gly/code/yolomm python -u trainMMrgbt.py >> recode/rgbt/yolo11n-mm-mid.txt 2>&1
# import os
# os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# from ultralytics import YOLOMM

# if __name__ == '__main__':
#     model = YOLOMM('/home/gly/code/yolomm/ultralytics/cfg/models/mm/detect/yolo11/base/yolo11n-mm-mid.yaml')
#     # model = YOLOMM('yolo11n-mm-mid.yaml')
#     model.train(
#                 # data='/root/datasets/M3FD_split/data.yaml',
#                 data='/home/gly/code/yolomm/data.yaml',
#                 epochs=100, batch=24, imgsz=640, close_mosaic=10,
#                 scale='n',  # 选择模型 YAML `scales`（n/s/m/l/x）；
#                 # fraction=0.1,
#                 # modality='rgb', # 模态消融参数 非必要不得开启（模态消融不等于单模态）
#                 # single_modal='rgb', #单模态参数的，单模态模型 yaml 设置之后可以跑对应模态的单模态训练
#                 # loss_cls='bce',      # 分类损失类型: bce | focal | efl | qfl (默认: bce)
#                 # loss_box='ciou',     # IoU 损失类型: iou | giou | diou | ciou | siou | eiou | wiou | alphaiou (默认: ciou)
#                 # loss_dfl=True,       # DFL 损失开关: True | False (默认: True)
#                 amp=True,  #启动amp混合精度加速，loss 出现 nan 或者无法运行时候可以尝试关闭？？
#                 # device=[0],
#                 cache='disk',
#                 device=[0,1],
#                 seed=0,
#                 exist_ok=False,
#                 workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#                 # afss_enabled=True, #启动afss训练采样机制
#                 project='runs/rgbt', name='yolo11n-mm-mid')




# #这是提供 参考示例的 YOLOMM训练器使用方法
# #按必须按照你的需求来更改而不是盲目去使用   PYTHONPATH=/home/gly/code/yolomm python -u trainMMrgbt.py >> recode/rgbt/yolo11n-mm-mid-p2.txt 2>&1
# import os
# os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# from ultralytics import YOLOMM

# if __name__ == '__main__':
#     model = YOLOMM('/home/gly/code/yolomm/ultralytics/cfg/models/mm/yolo11n-mm-mid-p2.yaml')
#     # model = YOLOMM('yolo11n-mm-mid.yaml')
#     model.train(
#                 # data='/root/datasets/M3FD_split/data.yaml',
#                 data='/home/gly/code/yolomm/data.yaml',
#                 epochs=100, batch=24, imgsz=640, close_mosaic=10,
#                 scale='n',  # 选择模型 YAML `scales`（n/s/m/l/x）；
#                 # fraction=0.1,
#                 # modality='rgb', # 模态消融参数 非必要不得开启（模态消融不等于单模态）
#                 # single_modal='rgb', #单模态参数的，单模态模型 yaml 设置之后可以跑对应模态的单模态训练
#                 # loss_cls='bce',      # 分类损失类型: bce | focal | efl | qfl (默认: bce)
#                 # loss_box='ciou',     # IoU 损失类型: iou | giou | diou | ciou | siou | eiou | wiou | alphaiou (默认: ciou)
#                 # loss_dfl=True,       # DFL 损失开关: True | False (默认: True)
#                 amp=True,  #启动amp混合精度加速，loss 出现 nan 或者无法运行时候可以尝试关闭？？
#                 # device=[0],
#                 cache='disk',
#                 device=[0,1],
#                 seed=0,
#                 exist_ok=False,
#                 workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#                 # afss_enabled=True, #启动afss训练采样机制
#                 project='runs/rgbt', name='yolo11n-mm-mid-p2')





# #这是提供 参考示例的 YOLOMM训练器使用方法
# #按必须按照你的需求来更改而不是盲目去使用   PYTHONPATH=/home/gly/code/yolomm python -u trainMMrgbt.py >> recode/rgbt/yolo11n-mm-mid-p2-rgbir.txt 2>&1
# import os
# os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# from ultralytics import YOLOMM

# if __name__ == '__main__':
#     model = YOLOMM('/home/gly/code/yolomm/ultralytics/cfg/models/mm/my/yolo11n-mm-mid-p2-rgbir.yaml')
#     # model = YOLOMM('yolo11n-mm-mid.yaml')
#     model.train(
#                 # data='/root/datasets/M3FD_split/data.yaml',
#                 data='/home/gly/code/yolomm/data.yaml',
#                 epochs=100, batch=24, imgsz=640, close_mosaic=10,
#                 scale='n',  # 选择模型 YAML `scales`（n/s/m/l/x）；
#                 # fraction=0.1,
#                 # modality='rgb', # 模态消融参数 非必要不得开启（模态消融不等于单模态）
#                 # single_modal='rgb', #单模态参数的，单模态模型 yaml 设置之后可以跑对应模态的单模态训练
#                 # loss_cls='bce',      # 分类损失类型: bce | focal | efl | qfl (默认: bce)
#                 # loss_box='ciou',     # IoU 损失类型: iou | giou | diou | ciou | siou | eiou | wiou | alphaiou (默认: ciou)
#                 # loss_dfl=True,       # DFL 损失开关: True | False (默认: True)
#                 amp=True,  #启动amp混合精度加速，loss 出现 nan 或者无法运行时候可以尝试关闭？？
#                 # device=[0],
#                 cache='disk',
#                 device=[0,1],
#                 seed=0,
#                 exist_ok=False,
#                 workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#                 # afss_enabled=True, #启动afss训练采样机制
#                 project='runs/rgbt', name='yolo11n-mm-mid-p2-rgbir')




# #这是提供 参考示例的 YOLOMM训练器使用方法
# #按必须按照你的需求来更改而不是盲目去使用   PYTHONPATH=/home/gly/code/yolomm python -u trainMMrgbt.py >> recode/rgbt/yolo11n-mm-mid-soep.txt 2>&1
# import os
# os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# from ultralytics import YOLOMM

# if __name__ == '__main__':
#     model = YOLOMM('/home/gly/code/yolomm/ultralytics/cfg/models/mm/detect/yolo11/neck/yolo11n-mm-mid-soep.yaml')
#     # model = YOLOMM('yolo11n-mm-mid.yaml')
#     model.train(
#                 # data='/root/datasets/M3FD_split/data.yaml',
#                 data='/home/gly/code/yolomm/data.yaml',
#                 epochs=100, batch=24, imgsz=640, close_mosaic=10,
#                 scale='n',  # 选择模型 YAML `scales`（n/s/m/l/x）；
#                 # fraction=0.1,
#                 # modality='rgb', # 模态消融参数 非必要不得开启（模态消融不等于单模态）
#                 # single_modal='rgb', #单模态参数的，单模态模型 yaml 设置之后可以跑对应模态的单模态训练
#                 # loss_cls='bce',      # 分类损失类型: bce | focal | efl | qfl (默认: bce)
#                 # loss_box='ciou',     # IoU 损失类型: iou | giou | diou | ciou | siou | eiou | wiou | alphaiou (默认: ciou)
#                 # loss_dfl=True,       # DFL 损失开关: True | False (默认: True)
#                 amp=False,  #启动amp混合精度加速，loss 出现 nan 或者无法运行时候可以尝试关闭？？
#                 find_unused_parameters=True,     # 2. 👈 核心修复：允许模型中存在未参与loss计算的参数
#                 # device=[0],
#                 cache='disk',
#                 device=[0,1],
#                 seed=0,
#                 exist_ok=False,
#                 workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#                 # afss_enabled=True, #启动afss训练采样机制
#                 project='runs/rgbt', name='yolo11n-mm-mid-soep')




# #这是提供 参考示例的 YOLOMM训练器使用方法
# #按必须按照你的需求来更改而不是盲目去使用   PYTHONPATH=/home/gly/code/yolomm python -u trainMMrgbt.py >> recode/rgbt/yolo11n-mm-mid-soep-rfpn.txt 2>&1
# import os
# os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# from ultralytics import YOLOMM

# if __name__ == '__main__':
#     model = YOLOMM('/home/gly/code/yolomm/ultralytics/cfg/models/mm/detect/yolo11/neck/yolo11n-mm-mid-soep-rfpn.yaml')
#     # model = YOLOMM('yolo11n-mm-mid.yaml')
#     model.train(
#                 # data='/root/datasets/M3FD_split/data.yaml',
#                 data='/home/gly/code/yolomm/data.yaml',
#                 epochs=100, batch=24, imgsz=640, close_mosaic=10,
#                 scale='n',  # 选择模型 YAML `scales`（n/s/m/l/x）；
#                 # fraction=0.1,
#                 # modality='rgb', # 模态消融参数 非必要不得开启（模态消融不等于单模态）
#                 # single_modal='rgb', #单模态参数的，单模态模型 yaml 设置之后可以跑对应模态的单模态训练
#                 # loss_cls='bce',      # 分类损失类型: bce | focal | efl | qfl (默认: bce)
#                 # loss_box='ciou',     # IoU 损失类型: iou | giou | diou | ciou | siou | eiou | wiou | alphaiou (默认: ciou)
#                 # loss_dfl=True,       # DFL 损失开关: True | False (默认: True)
#                 amp=False,  #启动amp混合精度加速，loss 出现 nan 或者无法运行时候可以尝试关闭？？
#                 find_unused_parameters=True,     # 2. 👈 核心修复：允许模型中存在未参与loss计算的参数
#                 # device=[0],
#                 cache='disk',
#                 device=[0,1],
#                 seed=0,
#                 exist_ok=False,
#                 workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#                 # afss_enabled=True, #启动afss训练采样机制
#                 project='runs/rgbt', name='yolo11n-mm-mid-soep-rfpn')




# #这是提供 参考示例的 YOLOMM训练器使用方法
# #按必须按照你的需求来更改而不是盲目去使用   PYTHONPATH=/home/gly/code/yolomm python -u trainMMrgbt.py >> recode/rgbt/yolo11n-mm-mid-p2-ir.txt 2>&1
# import os
# os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# from ultralytics import YOLOMM

# if __name__ == '__main__':
#     model = YOLOMM('/home/gly/code/yolomm/ultralytics/cfg/models/mm/my/yolo11n-mm-mid-p2-ir.yaml')
#     # model = YOLOMM('yolo11n-mm-mid.yaml')
#     model.train(
#                 # data='/root/datasets/M3FD_split/data.yaml',
#                 data='/home/gly/code/yolomm/data.yaml',
#                 epochs=100, batch=24, imgsz=640, close_mosaic=10,
#                 scale='n',  # 选择模型 YAML `scales`（n/s/m/l/x）；
#                 # fraction=0.1,
#                 # modality='rgb', # 模态消融参数 非必要不得开启（模态消融不等于单模态）
#                 # single_modal='rgb', #单模态参数的，单模态模型 yaml 设置之后可以跑对应模态的单模态训练
#                 # loss_cls='bce',      # 分类损失类型: bce | focal | efl | qfl (默认: bce)
#                 # loss_box='ciou',     # IoU 损失类型: iou | giou | diou | ciou | siou | eiou | wiou | alphaiou (默认: ciou)
#                 # loss_dfl=True,       # DFL 损失开关: True | False (默认: True)
#                 amp=True,  #启动amp混合精度加速，loss 出现 nan 或者无法运行时候可以尝试关闭？？
#                 # device=[0],
#                 cache='disk',
#                 device=[0,1],
#                 seed=0,
#                 exist_ok=False,
#                 workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#                 # afss_enabled=True, #启动afss训练采样机制
#                 project='runs/rgbt', name='yolo11n-mm-mid-p2-ir')




# #这是提供 参考示例的 YOLOMM训练器使用方法
# #按必须按照你的需求来更改而不是盲目去使用   PYTHONPATH=/home/gly/code/yolomm python -u trainMMrgbt.py >> recode/rgbt/yolo11n-mm-mid-p2rgb-p3ir.txt 2>&1
# import os
# os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# from ultralytics import YOLOMM

# if __name__ == '__main__':
#     model = YOLOMM('/home/gly/code/yolomm/ultralytics/cfg/models/mm/my/yolo11n-mm-mid-p2rgb-p3ir.yaml')
#     # model = YOLOMM('yolo11n-mm-mid.yaml')
#     model.train(
#                 # data='/root/datasets/M3FD_split/data.yaml',
#                 data='/home/gly/code/yolomm/data.yaml',
#                 epochs=100, batch=24, imgsz=640, close_mosaic=10,
#                 scale='n',  # 选择模型 YAML `scales`（n/s/m/l/x）；
#                 # fraction=0.1,
#                 # modality='rgb', # 模态消融参数 非必要不得开启（模态消融不等于单模态）
#                 # single_modal='rgb', #单模态参数的，单模态模型 yaml 设置之后可以跑对应模态的单模态训练
#                 # loss_cls='bce',      # 分类损失类型: bce | focal | efl | qfl (默认: bce)
#                 # loss_box='ciou',     # IoU 损失类型: iou | giou | diou | ciou | siou | eiou | wiou | alphaiou (默认: ciou)
#                 # loss_dfl=True,       # DFL 损失开关: True | False (默认: True)
#                 amp=True,  #启动amp混合精度加速，loss 出现 nan 或者无法运行时候可以尝试关闭？？
#                 # device=[0],
#                 cache='disk',
#                 device=[0,1],
#                 seed=0,
#                 exist_ok=False,
#                 workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#                 # afss_enabled=True, #启动afss训练采样机制
#                 project='runs/rgbt', name='yolo11n-mm-mid-p2rgb-p3ir')




# #这是提供 参考示例的 YOLOMM训练器使用方法
# #按必须按照你的需求来更改而不是盲目去使用   PYTHONPATH=/home/gly/code/yolomm python -u trainMMrgbt.py >> recode/rgbt/yolo11n-mm-mid-p2rgb-p3rgbir.txt 2>&1
# import os
# os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# from ultralytics import YOLOMM

# if __name__ == '__main__':
#     model = YOLOMM('/home/gly/code/yolomm/ultralytics/cfg/models/mm/my/yolo11n-mm-mid-p2rgb-p3rgbir.yaml')
#     # model = YOLOMM('yolo11n-mm-mid.yaml')
#     model.train(
#                 # data='/root/datasets/M3FD_split/data.yaml',
#                 data='/home/gly/code/yolomm/data.yaml',
#                 epochs=100, batch=24, imgsz=640, close_mosaic=10,
#                 scale='n',  # 选择模型 YAML `scales`（n/s/m/l/x）；
#                 # fraction=0.1,
#                 # modality='rgb', # 模态消融参数 非必要不得开启（模态消融不等于单模态）
#                 # single_modal='rgb', #单模态参数的，单模态模型 yaml 设置之后可以跑对应模态的单模态训练
#                 # loss_cls='bce',      # 分类损失类型: bce | focal | efl | qfl (默认: bce)
#                 # loss_box='ciou',     # IoU 损失类型: iou | giou | diou | ciou | siou | eiou | wiou | alphaiou (默认: ciou)
#                 # loss_dfl=True,       # DFL 损失开关: True | False (默认: True)
#                 amp=True,  #启动amp混合精度加速，loss 出现 nan 或者无法运行时候可以尝试关闭？？
#                 # device=[0],
#                 cache='disk',
#                 device=[0,1],
#                 seed=0,
#                 exist_ok=False,
#                 workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#                 # afss_enabled=True, #启动afss训练采样机制
#                 project='runs/rgbt', name='yolo11n-mm-mid-p2rgb-p3rgbir')





# #这是提供 参考示例的 YOLOMM训练器使用方法
# #按必须按照你的需求来更改而不是盲目去使用   PYTHONPATH=/home/gly/code/yolomm python -u trainMMrgbt.py >> recode/rgbt/yolo11n-mm-mid-p2rgb-p45-mcf.txt 2>&1
# import os
# os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# from ultralytics import YOLOMM

# if __name__ == '__main__':
#     model = YOLOMM('/home/gly/code/yolomm/ultralytics/cfg/models/mm/my/yolo11n-mm-mid-p2rgb-p45-mcf.yaml')
#     # model = YOLOMM('yolo11n-mm-mid.yaml')
#     model.train(
#                 # data='/root/datasets/M3FD_split/data.yaml',
#                 data='/home/gly/code/yolomm/data.yaml',
#                 epochs=100, batch=24, imgsz=640, close_mosaic=10,
#                 scale='n',  # 选择模型 YAML `scales`（n/s/m/l/x）；
#                 # fraction=0.1,
#                 # modality='rgb', # 模态消融参数 非必要不得开启（模态消融不等于单模态）
#                 # single_modal='rgb', #单模态参数的，单模态模型 yaml 设置之后可以跑对应模态的单模态训练
#                 # loss_cls='bce',      # 分类损失类型: bce | focal | efl | qfl (默认: bce)
#                 # loss_box='ciou',     # IoU 损失类型: iou | giou | diou | ciou | siou | eiou | wiou | alphaiou (默认: ciou)
#                 # loss_dfl=True,       # DFL 损失开关: True | False (默认: True)
#                 amp=True,  #启动amp混合精度加速，loss 出现 nan 或者无法运行时候可以尝试关闭？？
#                 # device=[0],
#                 cache='disk',
#                 device=[0,1],
#                 seed=0,
#                 exist_ok=False,
#                 workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#                 # afss_enabled=True, #启动afss训练采样机制
#                 project='runs/rgbt', name='yolo11n-mm-mid-p2rgb-p45-mcf')





# #这是提供 参考示例的 YOLOMM训练器使用方法
# #按必须按照你的需求来更改而不是盲目去使用   PYTHONPATH=/home/gly/code/yolomm python -u trainMMrgbt.py >> recode/rgbt/yolo11n-mm-mid-p2rgb-p45-mfm.txt 2>&1
# import os
# os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# from ultralytics import YOLOMM

# if __name__ == '__main__':
#     model = YOLOMM('/home/gly/code/yolomm/ultralytics/cfg/models/mm/my/yolo11n-mm-mid-p2rgb-p45-mfm.yaml')
#     # model = YOLOMM('yolo11n-mm-mid.yaml')
#     model.train(
#                 # data='/root/datasets/M3FD_split/data.yaml',
#                 data='/home/gly/code/yolomm/data.yaml',
#                 epochs=100, batch=24, imgsz=640, close_mosaic=10,
#                 scale='n',  # 选择模型 YAML `scales`（n/s/m/l/x）；
#                 # fraction=0.1,
#                 # modality='rgb', # 模态消融参数 非必要不得开启（模态消融不等于单模态）
#                 # single_modal='rgb', #单模态参数的，单模态模型 yaml 设置之后可以跑对应模态的单模态训练
#                 # loss_cls='bce',      # 分类损失类型: bce | focal | efl | qfl (默认: bce)
#                 # loss_box='ciou',     # IoU 损失类型: iou | giou | diou | ciou | siou | eiou | wiou | alphaiou (默认: ciou)
#                 # loss_dfl=True,       # DFL 损失开关: True | False (默认: True)
#                 amp=True,  #启动amp混合精度加速，loss 出现 nan 或者无法运行时候可以尝试关闭？？
#                 # device=[0],
#                 cache='disk',
#                 device=[0,1],
#                 seed=0,
#                 exist_ok=False,
#                 workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#                 # afss_enabled=True, #启动afss训练采样机制
#                 project='runs/rgbt', name='yolo11n-mm-mid-p2rgb-p45-mfm')





# #这是提供 参考示例的 YOLOMM训练器使用方法
# #按必须按照你的需求来更改而不是盲目去使用   PYTHONPATH=/home/gly/code/yolomm python -u trainMMrgbt.py >> recode/rgbt/yolo11n-mm-mid-p2rgb-logstem.txt 2>&1
# import os
# os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# from ultralytics import YOLOMM

# if __name__ == '__main__':
#     model = YOLOMM('/home/gly/code/yolomm/ultralytics/cfg/models/mm/my/yolo11n-mm-mid-p2rgb-logstem.yaml')
#     # model = YOLOMM('yolo11n-mm-mid.yaml')
#     model.train(
#                 # data='/root/datasets/M3FD_split/data.yaml',
#                 data='/home/gly/code/yolomm/data.yaml',
#                 epochs=100, batch=16, imgsz=640, close_mosaic=10,
#                 scale='n',  # 选择模型 YAML `scales`（n/s/m/l/x）；
#                 # fraction=0.1,
#                 # modality='rgb', # 模态消融参数 非必要不得开启（模态消融不等于单模态）
#                 # single_modal='rgb', #单模态参数的，单模态模型 yaml 设置之后可以跑对应模态的单模态训练
#                 # loss_cls='bce',      # 分类损失类型: bce | focal | efl | qfl (默认: bce)
#                 # loss_box='ciou',     # IoU 损失类型: iou | giou | diou | ciou | siou | eiou | wiou | alphaiou (默认: ciou)
#                 # loss_dfl=True,       # DFL 损失开关: True | False (默认: True)
#                 amp=False,  #启动amp混合精度加速，loss 出现 nan 或者无法运行时候可以尝试关闭？？
#                 # device=[0],
#                 cache='disk',
#                 device=[0,1],
#                 seed=0,
#                 exist_ok=False,
#                 workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#                 # afss_enabled=True, #启动afss训练采样机制
#                 project='runs/rgbt', name='yolo11n-mm-mid-p2rgb-logstem')






# #这是提供 参考示例的 YOLOMM训练器使用方法
# #按必须按照你的需求来更改而不是盲目去使用   PYTHONPATH=/home/gly/code/yolomm python -u trainMMrgbt.py >> recode/rgbt/yolo11n-mm-mid-p2rgb-logstem-dysample-3.txt 2>&1
# import os
# os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# from ultralytics import YOLOMM

# if __name__ == '__main__':
#     model = YOLOMM('/home/gly/code/yolomm/ultralytics/cfg/models/mm/my/yolo11n-mm-mid-p2rgb-logstem-dysample-3.yaml')
#     # model = YOLOMM('yolo11n-mm-mid.yaml')
#     model.train(
#                 # data='/root/datasets/M3FD_split/data.yaml',
#                 data='/home/gly/code/yolomm/data.yaml',
#                 epochs=100, batch=16, imgsz=640, close_mosaic=10,
#                 scale='n',  # 选择模型 YAML `scales`（n/s/m/l/x）；
#                 # fraction=0.1,
#                 # modality='rgb', # 模态消融参数 非必要不得开启（模态消融不等于单模态）
#                 # single_modal='rgb', #单模态参数的，单模态模型 yaml 设置之后可以跑对应模态的单模态训练
#                 # loss_cls='bce',      # 分类损失类型: bce | focal | efl | qfl (默认: bce)
#                 # loss_box='ciou',     # IoU 损失类型: iou | giou | diou | ciou | siou | eiou | wiou | alphaiou (默认: ciou)
#                 # loss_dfl=True,       # DFL 损失开关: True | False (默认: True)
#                 amp=False,  #启动amp混合精度加速，loss 出现 nan 或者无法运行时候可以尝试关闭？？
#                 # device=[0],
#                 cache='disk',
#                 device=[0,1],
#                 seed=0,
#                 exist_ok=False,
#                 workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#                 # afss_enabled=True, #启动afss训练采样机制
#                 project='runs/rgbt', name='yolo11n-mm-mid-p2rgb-logstem-dysample-3')







# #这是提供 参考示例的 YOLOMM训练器使用方法
# #按必须按照你的需求来更改而不是盲目去使用   PYTHONPATH=/home/gly/code/yolomm python -u trainMMrgbt.py >> recode/rgbt/yolo11n-mm-mid-p2rgb-logstem-p3rgbir.txt 2>&1
# import os
# os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# from ultralytics import YOLOMM

# if __name__ == '__main__':
#     model = YOLOMM('/home/gly/code/yolomm/ultralytics/cfg/models/mm/my/yolo11n-mm-mid-p2rgb-logstem-p3rgbir.yaml')
#     # model = YOLOMM('yolo11n-mm-mid.yaml')
#     model.train(
#                 # data='/root/datasets/M3FD_split/data.yaml',
#                 data='/home/gly/code/yolomm/data.yaml',
#                 epochs=100, batch=16, imgsz=640, close_mosaic=10,
#                 scale='n',  # 选择模型 YAML `scales`（n/s/m/l/x）；
#                 # fraction=0.1,
#                 # modality='rgb', # 模态消融参数 非必要不得开启（模态消融不等于单模态）
#                 # single_modal='rgb', #单模态参数的，单模态模型 yaml 设置之后可以跑对应模态的单模态训练
#                 # loss_cls='bce',      # 分类损失类型: bce | focal | efl | qfl (默认: bce)
#                 # loss_box='ciou',     # IoU 损失类型: iou | giou | diou | ciou | siou | eiou | wiou | alphaiou (默认: ciou)
#                 # loss_dfl=True,       # DFL 损失开关: True | False (默认: True)
#                 amp=False,  #启动amp混合精度加速，loss 出现 nan 或者无法运行时候可以尝试关闭？？
#                 # device=[0],
#                 cache='disk',
#                 device=[0,1],
#                 seed=0,
#                 exist_ok=False,
#                 workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#                 # afss_enabled=True, #启动afss训练采样机制
#                 project='runs/rgbt', name='yolo11n-mm-mid-p2rgb-logstem-p3rgbir')




# #这是提供 参考示例的 YOLOMM训练器使用方法
# #按必须按照你的需求来更改而不是盲目去使用   PYTHONPATH=/home/gly/code/yolomm python -u trainMMrgbt.py >> recode/rgbt/yolo11n-mm-mid-p2rgb-logstem-p3mjrnet.txt 2>&1
# import os
# os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# from ultralytics import YOLOMM

# if __name__ == '__main__':
#     model = YOLOMM('/home/gly/code/yolomm/ultralytics/cfg/models/mm/my/yolo11n-mm-mid-p2rgb-logstem-p3mjrnet.yaml')
#     # model = YOLOMM('yolo11n-mm-mid.yaml')
#     model.train(
#                 # data='/root/datasets/M3FD_split/data.yaml',
#                 data='/home/gly/code/yolomm/data.yaml',
#                 epochs=100, batch=16, imgsz=640, close_mosaic=10,
#                 scale='n',  # 选择模型 YAML `scales`（n/s/m/l/x）；
#                 # fraction=0.1,
#                 # modality='rgb', # 模态消融参数 非必要不得开启（模态消融不等于单模态）
#                 # single_modal='rgb', #单模态参数的，单模态模型 yaml 设置之后可以跑对应模态的单模态训练
#                 # loss_cls='bce',      # 分类损失类型: bce | focal | efl | qfl (默认: bce)
#                 # loss_box='ciou',     # IoU 损失类型: iou | giou | diou | ciou | siou | eiou | wiou | alphaiou (默认: ciou)
#                 # loss_dfl=True,       # DFL 损失开关: True | False (默认: True)
#                 amp=False,  #启动amp混合精度加速，loss 出现 nan 或者无法运行时候可以尝试关闭？？
#                 # device=[0],
#                 cache='disk',
#                 device=[0,1],
#                 seed=0,
#                 exist_ok=False,
#                 workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#                 # afss_enabled=True, #启动afss训练采样机制
#                 project='runs/rgbt', name='yolo11n-mm-mid-p2rgb-logstem-p3mjrnet')




# #这是提供 参考示例的 YOLOMM训练器使用方法
# #按必须按照你的需求来更改而不是盲目去使用   PYTHONPATH=/home/gly/code/yolomm python -u trainMMrgbt.py >> recode/rgbt/yolo11n-mm-mid-p2rgb-logstem-p3marf.txt 2>&1
# import os
# os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

# from ultralytics import YOLOMM

# if __name__ == '__main__':
#     model = YOLOMM('/home/gly/code/yolomm/ultralytics/cfg/models/mm/my/yolo11n-mm-mid-p2rgb-logstem-p3marf.yaml')
#     # model = YOLOMM('yolo11n-mm-mid.yaml')
#     model.train(
#                 # data='/root/datasets/M3FD_split/data.yaml',
#                 data='/home/gly/code/yolomm/data.yaml',
#                 epochs=100, batch=16, imgsz=640, close_mosaic=10,
#                 scale='n',  # 选择模型 YAML `scales`（n/s/m/l/x）；
#                 # fraction=0.1,
#                 # modality='rgb', # 模态消融参数 非必要不得开启（模态消融不等于单模态）
#                 # single_modal='rgb', #单模态参数的，单模态模型 yaml 设置之后可以跑对应模态的单模态训练
#                 # loss_cls='bce',      # 分类损失类型: bce | focal | efl | qfl (默认: bce)
#                 # loss_box='ciou',     # IoU 损失类型: iou | giou | diou | ciou | siou | eiou | wiou | alphaiou (默认: ciou)
#                 # loss_dfl=True,       # DFL 损失开关: True | False (默认: True)
#                 amp=False,  #启动amp混合精度加速，loss 出现 nan 或者无法运行时候可以尝试关闭？？
#                 # device=[0],
#                 cache='disk',
#                 device=[0,1],
#                 seed=0,
#                 exist_ok=False,
#                 workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
#                 # afss_enabled=True, #启动afss训练采样机制
#                 project='runs/rgbt', name='yolo11n-mm-mid-p2rgb-logstem-p3marf')



# 这是提供 参考示例的 YOLOMM训练器使用方法
# 按必须按照你的需求来更改而不是盲目去使用   
# PYTHONPATH=/home/gly/code/yolomm python -u trainMMrgbt.py >> recode/train/yolo11n-mm-mid.txt 2>&1
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0,1')  # GPU控制: 0/1/2/3 或 "0,1" 多卡

from ultralytics import YOLOMM

if __name__ == '__main__':
    model = YOLOMM('/home/gly/code/yolomm/my/yolo11n-mm-mid.yaml')
    # model = YOLOMM('yolo11n-mm-mid.yaml')
    model.train(
                # data='/root/datasets/M3FD_split/data.yaml',
                data='/home/gly/code/yolomm/data.yaml',
                epochs=100, batch=16, imgsz=640, close_mosaic=0,
                scale='n',  # 选择模型 YAML `scales`（n/s/m/l/x）；
                # fraction=0.1,
                # modality='rgb', # 模态消融参数 非必要不得开启（模态消融不等于单模态）
                # single_modal='rgb', #单模态参数的，单模态模型 yaml 设置之后可以跑对应模态的单模态训练
                # loss_cls='bce',      # 分类损失类型: bce | focal | efl | qfl (默认: bce)
                # loss_box='ciou',     # IoU 损失类型: iou | giou | diou | ciou | siou | eiou | wiou | alphaiou (默认: ciou)
                # loss_dfl=True,       # DFL 损失开关: True | False (默认: True)
                amp=False,  #启动amp混合精度加速，loss 出现 nan 或者无法运行时候可以尝试关闭？？
                # device=[0],
                cache='disk',
                device=[0,1],
                seed=0,
                exist_ok=False,
                workers=8, # 加载线程数，推荐 4/8，win 系统如果卡死就设置为 0 重试
                # afss_enabled=True, #启动afss训练采样机制
                project='runs/train', name='yolo11n-mm-mid')
