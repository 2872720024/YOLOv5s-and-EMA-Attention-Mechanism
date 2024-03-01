import math
import numpy as np
import argparse
import torch
import cv2
import pyzed.sl as sl
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_boxes, xyxy2xywh
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from utils.plots import Annotator
from threading import Lock, Thread
from utils.plots import colors
from time import sleep

#初始化线程变量
lock = Lock()
run_signal = False
exit_signal = False


def torch_thread(weights, img_size, conf_thres=0.6, iou_thres=0.9):
    global image_net, exit_signal, run_signal, detections, point_cloud

    #选择推理设备、设置图像尺寸
    device = select_device()
    half = device.type != 'cpu'
    imgsz = img_size

    #加载模型
    model = attempt_load(weights, device=device)
    names = model.names

    #获取所有层步幅并选择最大值作为模型步幅
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)
    if half:
        model.half()
    cudnn.benchmark = True

    #运行推理
    if device.type != 'cpu':
        #创建建一个大小为 (1, 3, imgsz, imgsz) 的零张量，并将其发送到GPU上，使用 type_as 方法将张量的数据类型设置为与模型的参数相同的数据类型。将该张量作为输入传递给模型进行推理。
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # 运行一次

    x = [0,0,0]
    y = [0,0,0]
    while not exit_signal:
        if run_signal:
            lock.acquire()
            #图像预处理
            img, ratio, pad = letterbox(image_net[:, :, :3], imgsz, auto=False)
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img = img / 255.0
            if len(img.shape) == 3:
                img = img[None]
            #模型推理
            pred = model(img, augment=False, visualize=False)[0]
            pred = non_max_suppression(pred, conf_thres, iou_thres)
            #打印检测框
            for i, det in enumerate(pred):          #遍历并获取索引i和对应结果det
                s, im0 = '', image_net.copy()           #初始化S为空字符串，复制image_net到im0
                gn = torch.tensor(image_net.shape)[[1, 0, 1, 0]]        #创建图片的形状信息张量
                annotator = Annotator(image_net, line_width=2, example=str('A'))  #实例化注释器
                if len(det):
                    det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], image_net.shape).round()        #缩放边界框坐标并赋值
                    #计算边界框坐标与对应距离并打印在图片上
                    for *xyxy, conf, cls in reversed(det): #反向遍历并赋值
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  #转化边界坐标并归一化
                        cent_x = round(xywh[0] * im0.shape[1])
                        cent_y = round(xywh[1] * im0.shape[0])
                        cent_w = round(xywh[2] * im0.shape[1])
                        point_1 = round(cent_x - 0.4 * cent_w)
                        point_2 = round(cent_x + 0.4 * cent_w)
                        wide_value_1 = point_cloud.get_value(point_1, cent_y)[1]
                        wide_value_2 = point_cloud.get_value(point_2, cent_y)[1]
                        try:
                            wide = round(wide_value_1[0], 4) - round(wide_value_2[0], 4)
                            wide = round(abs(wide * 1000))
                        except:
                            wide = 0.00
                            pass
                        point_cloud_value = point_cloud.get_value(cent_x, cent_y)[1]
                        point_cloud_value = point_cloud_value * -100.00
                        if point_cloud_value[2] > 0.00:
                            if names[int(cls)] == 'a':
                                x[0] = point_cloud_value[0]
                                y[0] = point_cloud_value[1]
                            if names[int(cls)] == 'b':
                                x[1] = point_cloud_value[0]
                                y[1] = point_cloud_value[1]
                            if names[int(cls)] == 'c':
                                x[2] = point_cloud_value[0]
                                y[2] = point_cloud_value[1]
                            try:
                                point_cloud_value[0] = round(point_cloud_value[0])
                                point_cloud_value[1] = round(point_cloud_value[1])
                                point_cloud_value[2] = round(point_cloud_value[2])
                                distance = math.sqrt(
                                    point_cloud_value[0] * point_cloud_value[0] + point_cloud_value[1] *
                                    point_cloud_value[1] +
                                    point_cloud_value[2] * point_cloud_value[2])

                                #print("dis:", distance)
                                if distance <=300 and names[int(cls)] == "person":
                                    print("有行人进入危险距离，请注意！")
                                txt = '{0} dis:{1}cm'.format(names[int(cls)], round(distance))
                                annotator.box_label(xyxy, txt, color=colors(cls,True))

                            except:
                                pass
                        im = annotator.result()
                        cv2.imshow('zed camera', im)
                        key = cv2.waitKey(10)
                        if key == ord('q'):
                            break
                        if x[0] !=0 and x[1] !=0 and x[2] !=0:
                            xab = x[0] - x[1]
                            xbc = x[1] - x[2]
                            xac = x[0] - x[2]
                            yab = y[0] - y[1]
                            ybc = y[1] - y[2]
                            yac = y[0] - y[2]
                            angle1 = np.arctan(yab / xab)
                            angle2 = np.arctan(ybc / xbc)
                            angle3 = np.arctan(yac / xac)
                            angle = (angle3 + angle2 + angle1) / 3
                            angle = np.rad2deg(angle)
                            print(angle)
            lock.release()
            run_signal = False
        sleep(0.01)


def main():
    global image_net, exit_signal, run_signal, detections, point_cloud

    capture_thread = Thread(target=torch_thread,
                            kwargs={

                                'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres})
    capture_thread.start()

    print("初始化zed相机中")
    #实例化zed相机
    zed = sl.Camera()

    input_type = sl.InputType()
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)

    # 设置初始化zed相机参数
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)    #创建初始化对象
    init_params.camera_resolution = sl.RESOLUTION.HD2K     # 设置相机分辨率为 HD720。
    init_params.coordinate_units = sl.UNIT.METER        #设置坐标单位为米
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # 设置深度模式为ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP  #SHEZ 坐标系为右手坐标系
    init_params.depth_maximum_distance = 5      #设置深度图像最大距离为5米

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)

    #检测zed相机是否加载
    if status != sl.ERROR_CODE.SUCCESS:
        print("无法检测到相机，请检查设置")
        exit()
    #创建用于存储图像和深度数据的类
    image_left_tmp = sl.Mat()

    print("zed相机初始化完毕")
    #实例化相机位置跟踪参数对象
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(positional_tracking_parameters)

    #储存创建物体检测参数的对象并传入zed相机
    obj_param = sl.ObjectDetectionParameters()
    #obj_param.detection_model = sl.DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    zed.enable_object_detection(obj_param)

    #创建储存检测信息的对象并传入相机位置跟踪参数
    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    # 创建储存数据的对象
    #point_cloud_render = sl.Mat()
    point_cloud = sl.Mat()
    image_left = sl.Mat()
    depth = sl.Mat()

    while True and not exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            #获取图像信息
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            lock.release()
            run_signal = True

            #另一线程进行目标检测中，阻塞当前线程
            while run_signal:
                sleep(0.001)

            # 等待检测结果
            lock.acquire()
            # 处理检测结果
            lock.release()
            zed.retrieve_objects(objects, obj_runtime_param)
            zed.retrieve_image(image_left, sl.VIEW.LEFT)


        else:
            exit_signal = True
    exit_signal = True
    zed.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='excavators.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.1, help='object confidence threshold')
    opt = parser.parse_args()
    with torch.no_grad():
        main()

