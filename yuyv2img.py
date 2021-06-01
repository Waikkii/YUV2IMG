import numpy as np
import cv2
import time
from tqdm import tqdm

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

def yuv2rgb422(y, u, v):
    """
    :param y: y分量
    :param u: u分量
    :param v: v分量
    :return: rgb格式数据以及r,g,b分量
    """

    rows, cols = y.shape[0], y.shape[1]
    print(rows)
    print(cols)
    
    # 创建r,g,b分量
    r = np.zeros((rows, cols), np.uint8)
    g = np.zeros((rows, cols), np.uint8)
    b = np.zeros((rows, cols), np.uint8)

    for i in range(rows):
        for j in range(int(cols/2)):
            # r[i, 2 * j] = max(0,min(255,y[i, 2 * j] + 1.402 * (v[i, j] - 128)))
            # g[i, 2 * j] = max(0,min(255,y[i, 2 * j] - 0.34414 * (u[i, j] - 128) - 0.71414 * (v[i, j] - 128)))
            # b[i, 2 * j] = max(0,min(255,y[i, 2 * j] + 1.772 * (u[i, j] - 128)))

            # r[i, 2 * j+1] = max(0,min(255,y[i, 2 * j+1] + 1.402 * (v[i, j] - 128)))
            # g[i, 2 * j+1] = max(0,min(255,y[i, 2 * j+1] - 0.34414 * (u[i, j] - 128) - 0.71414 * (v[i, j] - 128)))
            # b[i, 2 * j+1] = max(0,min(255,y[i, 2 * j+1] + 1.772 * (u[i, j] - 128)))
            r[i, 2 * j] = max(0,min(255,y[i, 2 * j] + 1.14 * (v[i, j] - 128)))
            g[i, 2 * j] = max(0,min(255,y[i, 2 * j] - 0.39 * (u[i, j] - 128) - 0.58 * (v[i, j] - 128)))
            b[i, 2 * j] = max(0,min(255,y[i, 2 * j] + 2.03 * (u[i, j] - 128)))

            r[i, 2 * j+1] = max(0,min(255,y[i, 2 * j+1] + 1.14 * (v[i, j] - 128)))
            g[i, 2 * j+1] = max(0,min(255,y[i, 2 * j+1] - 0.39 * (u[i, j] - 128) - 0.58 * (v[i, j] - 128)))
            b[i, 2 * j+1] = max(0,min(255,y[i, 2 * j+1] + 2.03 * (u[i, j] - 128)))

    rgb = cv2.merge([b, g, r])
    return rgb

def merge_YUV2RGB_v1(Y, U, V):
    """
    转换YUV图像为RGB格式（放大U、V）
    :param Y: Y分量图像
    :param U: U分量图像
    :param V: V分量图像
    :return: RGB格式图像
    """
    # Y分量图像比U、V分量图像大一倍，想要合并3个分量，需要先放大U、V分量和Y分量一样大小
    enlarge_U = cv2.resize(U, (0, 0), fx=2.0, fy=1.0, interpolation=cv2.INTER_CUBIC)
    enlarge_V = cv2.resize(V, (0, 0), fx=2.0, fy=1.0, interpolation=cv2.INTER_CUBIC)

    # 合并YUV3通道
    img_YUV = cv2.merge([Y, enlarge_U, enlarge_V])

    dst = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
    return dst

def rgb_save(data, num):
    for i in tqdm(range(len(data[0]))):
        rgb = merge_YUV2RGB_v1(data[0][i],data[1][i],data[2][i])
        #cv2.imwrite('yuv2bgr/{}.jpg'.format(i + 1), rgb)
        cv2.imwrite('yuv2bgr/{}.jpg'.format(str(num)), rgb)

def yuv_import(filename,dims,numfrm,startfrm):
    fp = open(filename,'rb')

    fp.seek(0, 2)  # 设置文件指针到文件流的尾部 + 偏移 0
    fp_end = fp.tell()  # 获取文件尾指针位置

    frame_size = dims[0] * dims[1] * 2# 一帧图像所含的像素个数
    num_frame_max = fp_end // frame_size  # 计算 YUV 文件包含图像数
    print("This yuv file has {} frame imgs!".format(num_frame_max))
    fp.seek(frame_size * startfrm, 0)  # 设置文件指针到文件流的起始位置 + 偏移 frame_size * startframe
    print("Extract imgs start frame is {}!".format(startfrm + 1))
    Y=[]
    U=[]
    V=[]
    rows, cols = dims[1], dims[0]
    print(rows)
    print(cols)
    Yt_1=np.zeros(shape=(rows,cols//2), dtype='uint8', order='C')
    Yt_2=np.zeros(shape=(rows,cols//2), dtype='uint8', order='C')
    Yt=np.zeros(shape=(rows,cols), dtype='uint8', order='C')
    Ut=np.zeros(shape=(rows,cols//2), dtype='uint8', order='C')
    Vt=np.zeros(shape=(rows,cols//2), dtype='uint8', order='C')
    if numfrm == 0:
        numfrm = num_frame_max
    for i in tqdm(range(numfrm)):
        for m in range(rows):
            for n in range(cols//2):
                
                Yt_1[m,n]=ord(fp.read(1))
                Ut[m,n]=ord(fp.read(1))
                Yt_2[m,n]=ord(fp.read(1))
                Vt[m,n]=ord(fp.read(1))
        for m in range(rows):
            for n in range(cols//2):
                Yt[m, 2*n]   = Yt_1[m,n]
                Yt[m, 2*n+1] = Yt_1[m,n]
        Y=Y+[Yt]
        U=U+[Ut]
        V=V+[Vt]
    fp.close()
    return (Y,U,V)

if __name__ == '__main__':
    start_time = time.time()
    num = 3
    path = "a"+str(num)+".yuv"
    data = yuv_import(filename = path, dims = (1280,720), numfrm = 1, startfrm= 0)#Y,U,V numfrm为0代表全部
    # im=Image.frombytes('L',(1280,720),data[0][1].tobytes())#data[i][j]，i表示YUV第几个，j表示第几+1帧
    # print(getsizeof(object, default))
    # im.show()
    rgb_save(data, num)
    end_time = time.time()
    print("RGB Save Success! Cost time is %.2fs." % (end_time - start_time))