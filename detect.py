"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""
import cv2
import sys
import time
import torch
import argparse
from pathlib import Path
import torch.backends.cudnn as cudnn

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())    # add yolov5/ to path  as_posix windows路径改成linux格式
from models.experimental import attempt_load
from utils.plots import colors, plot_one_box
from utils.datasets import LoadStreams, LoadImages
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box


@torch.no_grad()
def run(weights='yolov5s.pt',   # 权重文件的路径
        source='data/images',   # 测试图片的路径
        imgsz=640,              # 推理尺寸大小
        conf_thres=0.25,        # 置信度阈值
        iou_thres=0.45,         # NMS IoU阈值
        max_det=1000,           # 每张图片最多的检测个数
        device='',              # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,         # 是否显示结果
        save_txt=False,         # 将结果保存为txt
        save_conf=False,        # 将置信度保存在txt中
        save_crop=False,        # save cropped prediction boxes
        nosave=False,           # 不保留图片或视频
        classes=None,           # 需要过滤的类别
        agnostic_nms=False,     # class-agnostic NMS
        augment=False,          # 增广推理
        visualize=False,        # 可视化特征
        update=False,           # update all models
        project='runs/detect',  # 结果保存的目录
        name='exp',             # 结果保存目标的子目录
        exist_ok=False,         # 存在工作目录，不需要创建
        line_thickness=3,       # bounding box 画线粗细 (pixels)
        hide_labels=False,      # 隐藏label
        hide_conf=False,        # 隐藏置信度
        half=False,             # 使用FP16，半精度推理
        ):
    save_img = not nosave and not source.endswith('.txt')     # 保留推理产生的图片
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    # 初始化
    set_logging()
    device = select_device(device)    # 运行算力选择
    half &= device.type != 'cpu'      # 半精度推理只支持CUDA
    model = attempt_load(weights, map_location=device)  # 加载模型
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)             # 检查图片尺寸
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16
    # 第二阶段分类器（没用）
    classify = False
    if classify:
        modelc = load_classifier(name='resnet50', n=2)  # initialize
        modelc.load_state_dict(torch.load('resnet50.pt', map_location=device)['model']).to(device).eval()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # 加速固定尺寸图片的推理
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        bs = len(dataset)       # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
        bs = 1                  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs     # 视频使用
    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # 运行一次
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0                               # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # 推理
        t1 = time_synchronized()
        pred = model(img, augment=augment, visualize= increment_path(
            save_dir / 'features', mkdir=True) if visualize else False)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)  # 执行NMS
        t2 = time_synchronized()
        # 执行分类（没用）
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        # 处理检测结果
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)      # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            if len(det):
                # 将检测结果映射到im0的尺寸
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()                   # 每一类检测
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:          # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 中心点长宽表示
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)       # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=line_thickness)
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            # 打印时间
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)   # 显示图片
                cv2.waitKey(1)            # 停1毫秒
            # 保存检测后的照片
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)   # 图片
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
    print(f'Done. ({time.time() - t0:.3f}s)')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='权重文件的路径')
    parser.add_argument('--source', type=str, default='data/images', help='测试图片的路径')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='推理尺寸大小')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU阈值')
    parser.add_argument('--max-det', type=int, default=1000, help='每张图片最多的检测个数')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='是否显示结果')
    parser.add_argument('--save-txt', action='store_true', help='将结果保存为txt')
    parser.add_argument('--save-conf', action='store_true', help='将置信度保存在txt中')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='不保留图片或视频')
    parser.add_argument('--classes', nargs='+', type=int, help='需要过滤的类别')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='增广推理')
    parser.add_argument('--visualize', action='store_true', help='可视化特征')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='结果保存的目录')
    parser.add_argument('--name', default='exp', help='结果保存目标的子目录')
    parser.add_argument('--exist-ok', action='store_true', help='存在工作目录，不需要创建')
    parser.add_argument('--line-thickness', default=3, type=int, help=' bounding box 画线粗细 (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='隐藏label')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='隐藏置信度')
    parser.add_argument('--half', action='store_true', help='使用FP16，半精度推理')
    opt = parser.parse_args()
    return opt


def main(opt):
    print(colorstr('detect: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
