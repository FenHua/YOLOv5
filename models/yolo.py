"""YOLOv5-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add yolov5/ to path

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import make_divisible, check_file, set_logging
from utils.plots import feature_visualization
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None
logger = logging.getLogger(__name__)


# 检测头
class Detect(nn.Module):
    stride = None            # strides computed during build
    onnx_dynamic = False     # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):    # 检测头网络
        super(Detect, self).__init__()
        self.nc = nc                                               # 类别数
        self.no = nc + 5                                           # 每个anchor的输出
        self.nl = len(anchors)                                     # 使用的检测头个数（3个特征层）
        self.na = len(anchors[0]) // 2                             # 锚点的数量
        self.grid = [torch.zeros(1)] * self.nl                     # 初始化网格（3个不同尺度的网格）
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)     # anchors
        self.register_buffer('anchors', a)                         # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)        # 卷积输出结果
        self.inplace = inplace                                     # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # 推理结果
        for i in range(self.nl):
            x[i] = self.m[i](x[i])       # 卷积操作得到对应特征层推理结果
            bs, _, ny, nx = x[i].shape   # 例子：x(bs,255,20,20) to x(bs,3,20,20,85)  85=80+5（x，y，w，h，confidence）
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()  # 网络的预测结果（bs，3，20，20，85）
            if not self.training:
                # 测试阶段的推理（需要解码）
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)                       # 网格化
                y = x[i].sigmoid()                                                               # sigmoid
                if self.inplace:
                    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]       # xy
                    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]                   # wh
                else:
                    # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
                    xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                    wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
                    y = torch.cat((xy, wh, y[..., 4:]), -1)                                      # results
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])    # 生成网格
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()  # grid


class Model(nn.Module):
    # 模型配置文件，输入通道数，类别数
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg                       # 模型配置文件
        else:
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)     # 模型配置文件
        # 模型定义
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)     # 获取输入通道
        if nc and nc != self.yaml['nc']:
            logger.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc                           # 重写配置文件中的类别数
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)          # 如果存在anchor则重写anchor信息
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # 解析出model, 需要输出的list
        self.names = [str(i) for i in range(self.yaml['nc'])]              # 默认的names（没用）
        self.inplace = self.yaml.get('inplace', True)                      # 保持配置不变
        # 构建 strides, anchors
        m = self.model[-1]           # Detect()
        if isinstance(m, Detect):
            s = 256                  # 2x min stride
            m.inplace = self.inplace
            m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])   # forward
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # 初始化操作
        # 初始化 weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self.forward_augment(x)                # 增广推理TTA
        return self.forward_once(x, profile, visualize)   # 单尺度推理, train

    # TTA 推理
    def forward_augment(self, x):
        img_size = x.shape[-2:]      # 长, 宽
        s = [1, 0.83, 0.67]          # 尺度
        f = [None, 3, None]          # flips (2-ud, 3-lr)
        y = []                       # 输出
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self.forward_once(xi)[0]                    # forward
            yi = self._descale_pred(yi, fi, si, img_size)    # 对检测结果进行还原
            y.append(yi)
        # 输出增强后的检测结果
        return torch.cat(y, 1), None

    # 预测推理
    def forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []             # 输出
        for m in self.model:
            if m.f != -1:          # 如果不是来自上层的结果
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # 循环赋值，输入为上一层结果
            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
            x = m(x)               # run
            y.append(x if m.i in self.save else None)  # save output
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        if profile:
            logger.info('%.1fms total' % sum(dt))
        return x

    # 对检测结果按照原图变换执行反变换
    def _descale_pred(self, p, flips, scale, img_size):
        # 根据增强方式执行逆操作
        if self.inplace:
            p[..., :4] /= scale                      # 反缩放
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # 上下翻转反变换
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # 左右翻转反变换
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale
            if flips == 2:
                y = img_size[0] - y                  # 上下翻转反变换
            elif flips == 3:
                x = img_size[1] - x                  # 左右翻转反变换
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    # 网络初始化操作
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    # 输出偏置
    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # 将卷积和BN操作合并
    def fuse(self):
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # 更新卷积操作
                delattr(m, 'bn')                         # 去除BN层
                m.forward = m.fuseforward                # update forward
        self.info()
        return self

    # 添加或者不提交NMS操作
    def nms(self, mode=True):
        present = type(self.model[-1]) is NMS    # last layer is NMS
        if mode and not present:
            logger.info('Adding NMS... ')
            m = NMS()                            # module
            m.f = -1                             # from
            m.i = self.model[-1].i + 1           # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            logger.info('Removing NMS... ')
            self.model = self.model[:-1]         # remove
        return self

    # 对网络的输入进行包装
    def autoshape(self):  # add AutoShape module
        logger.info('Adding AutoShape... ')
        m = AutoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
        return m

    # 输出模型信息
    def info(self, verbose=False, img_size=640):
        model_info(self, verbose, img_size)


def parse_model(d, ch):
    # d: model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    # depth_multiple 表示bottleneckCSP模块的层缩放系数，width_multiple 表示channel的缩放系数
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # anchor数量
    no = na * (nc + 5)                                                     # 输出数量 = anchors * (classes + 5)
    layers, save, c2 = [], [], ch[-1]                                      # 网络层, 需要保存的中间结果, 输出通道
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):        # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a             # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n                          # 网络深度（当前模块的深度）
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR]:
            c1, c2 = ch[f], args[0]
            if c2 != no:                  # if not output
                c2 = make_divisible(c2 * gw, 8)
            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR]:
                args.insert(2, n)         # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')                                 # module type
        np = sum([x.numel() for x in m_.parameters()])                            # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)       # 检查配置文件
    set_logging()
    device = select_device(opt.device)  # 计算环境
    # Create model
    model = Model(opt.cfg).to(device)
    model.train()

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 320, 320).to(device)
    # y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # logger.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
