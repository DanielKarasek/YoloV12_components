import torch

from YoloV12.losses.detection_loss import DetectionLoss


def setup_detection_test_data():
    # ------------------------------------------------------------------------------
    # 1) Define shapes and hyperparameters
    # ------------------------------------------------------------------------------
    batch_size = 3
    layer_cnt  = 3  # e.g. YOLO typically predicts at multiple scales
    bin_count  = 5  # example
    cls_cnt    = 3  # example: number of classes
    W_list     = [16, 8, 4]  # output widths at each layer
    H_list     = [16, 8, 4]  # output heights at each layer
    Strides    = [1, 2, 4]  # example: strides at each layer

    # ------------------------------------------------------------------------------
    # 2) Create "predictions"
    #    We'll create a list of tensors, one per layer:
    #    predictions[i] shape = (batch_size, W_i, H_i, 4*bin_count + cls_cnt)
    # ------------------------------------------------------------------------------
    predictions = []
    for i in range(layer_cnt):
        W_i = W_list[i]
        H_i = H_list[i]
        predictions.append(
            torch.randn(batch_size, W_i, H_i, 4*bin_count + cls_cnt)/100

        )

    # ------------------------------------------------------------------------------
    # 3) Create "targets"
    #    keys = {0, 1, 2} for our 3 images
    # ------------------------------------------------------------------------------
    targets = {}

    # Image 0: zero bounding boxes
    targets[2] = {
        "img_idx": torch.empty((0,)),  # just the image index
        "gt_bboxes": torch.empty((0, 4)),  # shape = (0,4)
        "gt_labels": torch.empty((0,), dtype=torch.long)  # no labels
    }

    # Image 1: one bounding box (e.g. [x1, y1, x2, y2] in normalized coords)
    targets[1] = {
        "img_idx": torch.tensor([1]),  # just the image index
        "gt_bboxes": torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
        "gt_labels": torch.tensor([1])  # single label
    }

    # Image 2: three bounding boxes
    targets[0] = {
        "img_idx": torch.tensor([0, 0, 0]),  # just the image index
        "gt_bboxes": torch.tensor([
            [0.05, 0.15, 0.10, 0.30],
            [0.25, 0.35, 0.40, 0.60],
            [0.50, 0.55, 0.30, 0.25]
        ]),
        "gt_labels": torch.tensor([0, 1, 2])  # example class indices
    }

    # ------------------------------------------------------------------------------
    # Print shapes just to confirm
    # ------------------------------------------------------------------------------
    for i, pred in enumerate(predictions):
        print(f"predictions[{i}] shape:", pred.shape)

    for i in range(batch_size):
        gt_bboxes = targets[i]["gt_bboxes"]
        gt_labels = targets[i]["gt_labels"]
        print(f"Image {i}: gt_bboxes shape: {gt_bboxes.shape}, gt_labels: {gt_labels}")

    # ------------------------------------------------------------------------------
    # Stack targets into single dictionary
    # ------------------------------------------------------------------------------
    targets = {
        "img_idx": torch.cat([targets[i]["img_idx"] for i in range(batch_size)]),
        "gt_bboxes": torch.cat([targets[i]["gt_bboxes"] for i in range(batch_size)]),
        "gt_labels": torch.cat([targets[i]["gt_labels"] for i in range(batch_size)])
    }

    # ------------------------------------------------------------------------------
    # Setup the DetectionLoss object
    # ------------------------------------------------------------------------------

    detection_loss = DetectionLoss(bin_count,
                                   cls_cnt,
                                   Strides,
                                   1,
                                   2,
                                   3)
    return predictions, targets, detection_loss

if __name__ == "__main__":
    predictions, targets, detection_loss = setup_detection_test_data()
    loss = detection_loss(predictions, targets)
