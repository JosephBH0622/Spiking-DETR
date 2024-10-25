import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util_my.box_ops_my import box_cxcywh_to_xyxy, generalized_box_iou


# 匈牙利匹配
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        # bs is batch_size num_queries 是配置的每张图片中多少个目标
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # 前两个拉平在一起，最后一个维度是类别，进行softmax
        # [batch_size * num_queries, num_classes]
        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(1)  # 在最后个维度上进行softmax计算概率

        # [batch_size * num_queries, 4]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)

        # Also concat the target labels and boxes
        # targets: [{"labels": [num_target_boxes], "boxes": [num_target_boxes, 4]}, ..., {...}] bs个字典
        tgt_ids = torch.cat([v["labels"] for v in targets])  # target label [bs张图上的框数总和]
        tgt_bbox = torch.cat([v["boxes"] for v in targets])  # target bbox []

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # 取出对应的类别的分数，每个框 的 所有gt的label的分数 [T,bs*100,92][:,tgt_ids] --> [T,bs*100, 所有bs中img上的gt的数量和]
        # 一个预测框上的 对应的 在所有的gt上的分数
        cost_class = -out_prob[:, tgt_ids]

        # out_bbox is [bs*100,4] tgt_box is [all_img_gt_count,4]
        # Compute the L1 cost between boxes -> [bs*100, all_img_gt_count]
        # p=1 计算l1距离
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)  # cost_bbox 和 cost_class的维度是一样的

        # 计算的GIoU
        # 先进行中心点宽高变成坐上右下四个坐标值
        # 所有的框，跟gt的giou [bs*100, all_img_gt_count]
        # Compute the giou cost betwen boxes
        # 完全相同位置的框 giou是1，完全不相交的框，giou是负数
        # 因此这里加了一个负号，完全不相交的框的值就变成了整数，表示了更大的代价
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # 最终的代价矩阵，匈牙利匹配使用的，最终是要总的分配的代价最小
        # 前面都是各个项的权重系数
        # Final cost matrix [T, bs*100, all_img_gt_count]
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou

        # [bs*100, all_img_gt_count] -> [bs,100,all_img_gt_count]
        # .cpu 为了给scipy计算 维度变为 T, batch_size, 100, gt的数量
        C = C.view(bs, num_queries, -1).cpu()

        # 每个图片对应的gt的数量
        sizes = [len(v["boxes"]) for v in targets]

        # linear_sum_assignment 就是匈牙利算法
        # C.split 按照每个图片的gt的数量进行切分
        # 第一个值是100内的id，表明100内取哪一个框，第二个应该是对应了哪一个gt的id
        # indices is a list whose length is bs
        # indices的一个例子[(array([0, 51]), array([0, 1])), (array([13, 24, 54, 86]), array([0, 1, 3, 2]))]
        indices = [linear_sum_assignment(c[:, i]) for i, c in enumerate(C.split(sizes, -1))]

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
