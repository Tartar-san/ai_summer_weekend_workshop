def rect_to_bb(rect):
    """ Convert dlib rectangle to opencv-like bounding box """
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h


def crop_bb(img, bb):
    """ Take a bounding box and crop it from original image """
    x, y, w, h = bb
    cropped = img[y:y + h, x:x + w]
    return cropped


