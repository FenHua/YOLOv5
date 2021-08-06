# Loss functions
import torch
import torch.nn as nn
from utils.metrics import bbox_iou
from utils.torch_utils import is_parallel


# smooth 二分类交叉熵损失权重
def smooth_BCE(eps=0.1):
    # 返回正负类别平滑的交叉熵
    return 1.0 - 0.5 * eps, 0.5 * eps


# 二分类交叉熵（带logits）损失
class BCEBlurWithLogitsLoss(nn.Module):
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # logits二分类交叉熵损失
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)   # 计算损失
        pred = torch.sigmoid(pred)         # 根据logits计算概率
        dx = pred - true                   # 减少无标注的影响
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))  # 平衡因子
        loss *= alpha_factor
        # 适当削弱没有标注目标的影响
        return loss.mean()


# focal loss
class FocalLoss(nn.Module):
    # 将其它损失函数包装进Focal损失
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn             # 必须是 nn.BCEWithLogitsLoss()
        self.gamma = gamma                   # gamma系数用于调整难样本的系数
        self.alpha = alpha                   # alpha系数用于调整样本之间的平衡
        self.reduction = loss_fcn.reduction  # 聚合规则sum/mean
        self.loss_fcn.reduction = 'none'     # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)     # 计算loss
        pred_prob = torch.sigmoid(pred)      # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)   # 样本数平衡系数
        modulating_factor = (1.0 - p_t) ** self.gamma                      # 难分样本权重加大
        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


# QFL损失，FL只支持0、1这样的离散类别label，QFL可以处理smooth的label(label)
class QFocalLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn                 # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma                       # 调节样本难易程度
        self.alpha = alpha                       # 平衡正负样本
        self.reduction = loss_fcn.reduction      # 损失聚合方式
        self.loss_fcn.reduction = 'none'         # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)         # 损失
        pred_prob = torch.sigmoid(pred)          # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)  # 平衡正负样本比例
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma     # focus 难分样本
        loss *= alpha_factor * modulating_factor
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


# loss类
class ComputeLoss:
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        self.sort_obj_iou = False                 # 不对iou结果排序
        device = next(model.parameters()).device  # 得到计算环境
        h = model.hyp                             # 模型参数设置
        # 定义损失函数
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # 是否对label采取平滑
        g = h['fl_gamma']    # focal loss 难样本平衡系数 gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
        det = model.module.model[-1] if is_parallel(model) else model.model[-1]        # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, .02])   # P3-P7  特征点数比例p3：p4：...p7
        self.ssi = list(det.stride).index(16) if autobalance else 0                    # stride 16 index
        # 分类损失，置信度损失（前景），前景权重，超参，不同特征平衡权重
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):
        device = targets.device
        # 创建用来保存三层特征图的损失
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)     # 构建符合计算的真实标签
        # 对每个特征图进行计算损失
        for i, pi in enumerate(p):
            b, a, gj, gi = indices[i]  # 获取该层特征图上的真实信息：图像序号，anchor配置信息，位于特征图上的网格坐标
            tobj = torch.zeros_like(pi[..., 0], device=device)  # 存放真实信息中目标的置信度
            n = b.shape[0]             # 目标数
            # 存在bbox时计算分类回归损失，否则只计算置信度损失
            if n:
                ps = pi[b, a, gj, gi]  # 获取真值对应位置的预测box信息
                # 预测的bbox与真实bbox的损失
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # 预测的bbox结果
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()       # iou loss
                # 目标置信度预测结果
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # 利用IOU对预测置信度进行加权处理
                # 分类
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets 全零
                    t[range(n), tcls[i]] = self.cp                          # 正样本处为1
                    lcls += self.BCEcls(ps[:, 5:], t)                       # BCE 只计算正样本的损失

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)  # 获取置信度损失
            lobj += obji * self.balance[i]        # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]          # batch size
        loss = lbox + lobj + lcls
        return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]               # 锚框的数量, bbox的数量
        tcls, tbox, indices, anch = [], [], [], []       # 真实类别，真实box，标签详细的索引（图片以及网格坐标），对应的锚框
        # 将targets由(n*7)变成（3*n*7） 3个anchor，每个anchor都对应所有的box标签
        gain = torch.ones(7, device=targets.device)
        ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # anchor部署
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)              # append anchor indices
        g = 0.5  # 偏差
        off = torch.tensor([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                            ], device=targets.device).float() * g                       # offsets
        for i in range(self.nl):
            anchors = self.anchors[i]                           # 取不同特征图对应的anchors
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # 获取该层特征图的尺寸（1 1 w h w h 1）
            t = targets * gain  # 将box坐标转换到特征图上（在box生成标签时，对box坐标进行了归一化，即除以图像的宽高）
            # 通过对归一化的box乘以特征图的尺度，从而完成box坐标投影到特征图上
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]                        # wh ratio
                j = torch.max(r, 1. / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]                                                   # 去掉部分不相符的anchor
                # Offsets
                gxy = t[:, 2:4]           # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1. < g) & (gxy > 1.)).T
                l, m = ((gxi % 1. < g) & (gxi > 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))         # 得到四周的偏差
                t = t.repeat((5, 1, 1))[j]                                # 过滤
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j] # 生成所有box对应的偏置，原始box偏置为0
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T        # 得到图片id, 类别
            gxy = t[:, 2:4]                 # 当前网格下的xy
            gwh = t[:, 4:6]                 # 当前网格下的wh
            gij = (gxy - offsets).long()    # 获取每个box所在的网格的坐标
            gi, gj = gij.T                  # long（）取整

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # 保存图像序号，anchor序号，网格点坐标
            tbox.append(torch.cat((gxy - gij, gwh), 1))                                   # 获取(x,y)相对网格点的偏置以及box的宽高
            anch.append(anchors[a])                                                       # anchors
            tcls.append(c)                                                                # class
        return tcls, tbox, indices, anch