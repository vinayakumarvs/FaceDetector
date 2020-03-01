import cv2


def make_video(outvid, images=None, fps=30, size=None, is_color=True, format="FMP4"):
    """
    Create a video from a list of images.

    @param      outvid      output video
    @param      images      list of images to use in the video
    @param      fps         frame per second
    @param      size        size of each frame
    @param      is_color    color
    @param      format      see http://www.fourcc.org/codecs.php
    @return                 see http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html

    The function relies on http://opencv-python-tutroals.readthedocs.org/en/latest/.
    By default, the video will have the size of the first image.
    It will resize every image to this size before adding them to the video.
    """
    from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
    import os

    fourcc = VideoWriter_fourcc(*format)
    vid = None
    for image in images:
        if not os.path.exists(image):
            raise FileNotFoundError(image)
        img = imread(image)
        if vid is None:
            if size is None:
                size = img.shape[1], img.shape[0]
            vid = VideoWriter(outvid, fourcc, float(fps), size, is_color)
        if size[0] != img.shape[1] and size[1] != img.shape[0]:
            img = resize(img, size)
        vid.write(img)
    vid.release()
    return vid


def make_video_from_image_directory(images_dir, output_video_dir, video_file_name, fps=30):
    """
        Create a video from a list of images.

        @:param     images_dir          Input images directory
        @:param     output_video_dir    Output video directory
        @:param     video_file_name     File Name.
        @:param     fps                 Frames per sec
        @:return    output_video_path   Out out video path
    """
    import glob
    import os

    # Directory of images to run detection on
    images = list(glob.iglob (os.path.join (images_dir, '*.jpg')))

    # Sort the images by integer index
    images = sorted(images, key=lambda x: float (os.path.split (x)[1].split ("_")[0]))

    output_video_path = os.path.join(output_video_dir, video_file_name)
    make_video(output_video_path, images, fps=30)

    return output_video_path


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    """
    Concatenate frames/images of different sizes horizontally
    :param im_list: frame/image list
    :param interpolation: interpolation
    :return: horizontally concatenated frame
    """
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation) for im in im_list]
    return cv2.hconcat(im_list_resize)


def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    """
    Concatenate frames/images of different sizes vertically
    :param im_list: frame/image list
    :param interpolation: interpolation
    :return: vertically concatenated frame
    """
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation) for im in im_list]
    return cv2.vconcat(im_list_resize)


def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
    """
    Concatenate frames/images in 2d space based on the input 2d array
    :param im_list_2d: frames/images in a 2d array
    :param interpolation: interpolation
    :return: final concatenated image
    """
    im_list_v = [hconcat_resize_min(im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
    return vconcat_resize_min(im_list_v, interpolation=cv2.INTER_CUBIC)