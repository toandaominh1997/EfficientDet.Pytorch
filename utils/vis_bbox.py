import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def vis_bbox(img, bbox, label=None, score=None,
             instance_colors=None, alpha=1., linewidth=2., ax=None):
    """Visualize bounding boxes inside the image.
    Args:
        img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`. If this is :obj:`None`, no image is displayed.
        bbox (~numpy.ndarray): An array of shape :math:`(R, 4)`, where
            :math:`R` is the number of bounding boxes in the image.
            Each element is organized
            by :math:`(y_{min}, x_{min}, y_{max}, x_{max})` in the second axis.
        label (~numpy.ndarray): An integer array of shape :math:`(R,)`.
            The values correspond to id for label names stored in
            :obj:`label_names`. This is optional.
        score (~numpy.ndarray): A float array of shape :math:`(R,)`.
             Each value indicates how confident the prediction is.
             This is optional.
        label_names (iterable of strings): Name of labels ordered according
            to label ids. If this is :obj:`None`, labels will be skipped.
        instance_colors (iterable of tuples): List of colors.
            Each color is RGB format and the range of its values is
            :math:`[0, 255]`. The :obj:`i`-th element is the color used
            to visualize the :obj:`i`-th instance.
            If :obj:`instance_colors` is :obj:`None`, the red is used for
            all boxes.
        alpha (float): The value which determines transparency of the
            bounding boxes. The range of this value is :math:`[0, 1]`.
        linewidth (float): The thickness of the edges of the bounding boxes.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.
    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.
    from: https://github.com/chainer/chainercv
    """

    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    if ax is None:
        fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        h, w, _ = img.shape
        w_ = w / 60.0
        h_ = w_ * (h / w)
        fig.set_size_inches((w_, h_))
        ax = plt.axes([0, 0, 1, 1])
    ax.imshow(img.astype(np.uint8))
    ax.axis('off')
    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return fig, ax

    if instance_colors is None:
        # Red
        instance_colors = np.zeros((len(bbox), 3), dtype=np.float32)
        instance_colors[:, 0] = 51
        instance_colors[:, 1] = 51
        instance_colors[:, 2] = 224
    instance_colors = np.array(instance_colors)

    for i, bb in enumerate(bbox):
        xy = (bb[0], bb[1])
        height = bb[3] - bb[1]
        width = bb[2] - bb[0]
        color = instance_colors[i % len(instance_colors)] / 255
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False,
            edgecolor=color, linewidth=linewidth, alpha=alpha))

        caption = []
        caption.append(label[i])
        if(len(score) > 0):
            sc = score[i]
            caption.append('{}'.format(sc))

        if len(caption) > 0:
            face_color = np.array([225, 51, 123])/255
            ax.text(bb[0], bb[1],
                    ': '.join(caption),
                    fontsize=12,
                    color='black',
                    style='italic',
                    bbox={'facecolor': face_color, 'edgecolor': face_color, 'alpha': 1, 'pad': 0})
    return fig, ax


if __name__ == '__main__':
    img = cv2.imread('./../docs/output.png')
    print('img: ', img.shape)
    img = np.array(img)
    # img = img.convert('RGB')
    bbox = np.array([[50, 50, 200, 200]])
    label = np.array(['toan'])
    score = np.array([100])
    ax, fig = vis_bbox(img=img,
                       bbox=bbox,
                       label=label,
                       score=score,
                       label_names=label_names
                       )
    fig.savefig('kaka.png')
    fig.show()
    plt.show()
