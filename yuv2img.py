import numpy as np
import cv2
from PIL import Image
def read_yuv444p_frame(file_name, dims):
    with open(file_name, "rb") as f:
        h, w = dims
        buffer = f.read(h * w * 3)
        img = np.frombuffer(buffer, dtype=np.uint8).reshape((3, h, w)).transpose((1, 2, 0))
        return img
# h = 720
# w = 1280
# yuv444p_file_path = "D:\FDM Downloads\data\excel_01.yuv"
# yuv444p_img = read_yuv444p_frame(yuv444p_file_path, (h, w))
# cv2.imwrite("yuv444p-to-bgr.png", cv2.cvtColor(yuv444p_img, cv2.COLOR_YUV2BGR))

def yuv2bgr(file_name, height, width, start_frame):
    """
    :param file_name: 待处理 YUV 视频的名字
    :param height: YUV 视频中图像的高
    :param width: YUV 视频中图像的宽
    :param start_frame: 起始帧
    :return: None
    """
    fp = open(file_name, 'rb')
    fp.seek(0, 2)  # 设置文件指针到文件流的尾部 + 偏移 0
    fp_end = fp.tell()  # 获取文件尾指针位置

    frame_size = height * width * 3  # 一帧图像所含的像素个数
    num_frame = fp_end // frame_size  # 计算 YUV 文件包含图像数
    print("This yuv file has {} frame imgs!".format(num_frame))
    fp.seek(frame_size * start_frame, 0)  # 设置文件指针到文件流的起始位置 + 偏移 frame_size * startframe
    print("Extract imgs start frame is {}!".format(start_frame + 1))

    for i in range(num_frame - start_frame):
        yyyy_uv = np.zeros(shape=frame_size, dtype='uint8', order='C')
        for j in range(frame_size):
            yyyy_uv[j] = ord(fp.read(1))  # 读取 YUV 数据，并转换为 unicode

        img = yyyy_uv.reshape((3, height, width)).transpose((1, 2, 0)).astype('uint8')
        bgr_img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)  # 由于 opencv 不能直接读取 YUV 格式的文件, 所以要转换一下格式，支持的转换格式可参考资料 5
        cv2.imwrite('yuv2bgr/{}.jpg'.format(i + 1), bgr_img)  # 改变后缀即可实现不同格式图片的保存(jpg/bmp/png...)
        print("Extract frame {}".format(i + 1))

    fp.close()
    print("job done!")
    return None


def yuv_import(filename,dims,numfrm,startfrm):
    fp = open(filename,'rb')

    fp.seek(0, 2)  # 设置文件指针到文件流的尾部 + 偏移 0
    fp_end = fp.tell()  # 获取文件尾指针位置

    frame_size = dims[0] * dims[1] * 3  # 一帧图像所含的像素个数
    num_frame_max = fp_end // frame_size  # 计算 YUV 文件包含图像数
    print("This yuv file has {} frame imgs!".format(num_frame_max))
    fp.seek(frame_size * startfrm, 0)  # 设置文件指针到文件流的起始位置 + 偏移 frame_size * startframe
    print("Extract imgs start frame is {}!".format(startfrm + 1))
    Y=[]
    U=[]
    V=[]
    print(dims[0])
    print(dims[1])
    Yt=np.zeros(shape=(dims[0],dims[1]), dtype='uint8', order='C')
    Ut=np.zeros(shape=(dims[0],dims[1]), dtype='uint8', order='C')
    Vt=np.zeros(shape=(dims[0],dims[1]), dtype='uint8', order='C')
    if numfrm == 0:
        numfrm = num_frame_max
    for i in range(numfrm):
        for m in range(dims[0]):
            for n in range(dims[1]):
                #print m,n
                Yt[m,n]=ord(fp.read(1))
        for m in range(dims[0]):
            for n in range(dims[1]):
                Ut[m,n]=ord(fp.read(1))
        for m in range(dims[0]):
            for n in range(dims[1]):
                Vt[m,n]=ord(fp.read(1))
        Y=Y+[Yt]
        U=U+[Ut]
        V=V+[Vt]
    fp.close()
    return (Y,U,V)

if __name__ == '__main__':
    path = "D:\FDM Downloads\data\excel_01.yuv"
    data = yuv_import(filename = path, dims = (1280,720), numfrm = 1, startfrm= 0)#Y,U,V numfrm为0代表全部
    im=Image.frombytes('L',(1280,720),data[0][0].tobytes())#data[i][j]，i表示YUV第几个，j表示第几帧
    # print(getsizeof(object, default))
    im.show()
    # yuv2bgr(file_name=path, height=720, width=1280, start_frame=0)