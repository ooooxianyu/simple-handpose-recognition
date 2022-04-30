# simple-handpose-recognition

```
     ███████╗ ██████╗ ████████╗ ██╗   ██╗
     ██╔════╝   ██╔═╝ ██╔═════╝ ██║   ██║
     █████║     ██║   ████████║ ████████║
     ██╔══╝     ██║   ╚════╗██║ ██╔═══██║
     ██║      ██████╗ ████████║ ██║   ██║
     ╚═╝      ╚═════╝ ╚═══════╝ ╚═╝   ╚═╝
```
## 简介
该项目源自于EricLee开源的dpcas组件:https://gitcode.net/EricLee/dpcas;

由于作者后续加了很多模块，而我之前写的文章是关于手势识别的，所以单独把手势识别的模块拎了出来。并且在作者原先的代码上加入静态手势识别的分类模型。具体实现方式也可以参照我博客上的说明https://blog.csdn.net/weixin_41809530/article/details/122045068?spm=1001.2014.3001.5501；

动态手势识别的部分除了作者EricLee原先的点击状态，我本来想加入指尖轨迹跟踪和左右滑动的手势，但由于这阵子比较忙，只完成了一半，没时间继续完成，有兴趣的同学可以自己尝试。

# 手势识别实战模型

## 1.模型介绍
    --手部检测：yolov3
    --手指关键点模型：ReXnetV1
    --手势分类模型：基于resnet18的分类模型：https://github.com/weiaicunzai/pytorch-cifar100

## 2. 数据/权重/环境
    --相关权重和数据我放在百度云上自取，你需要把他按cfg里面的路径放在自己的项目中。 链接：https://pan.baidu.com/s/1jr-w6e1PVi3V9YJWxIww4w  提取码：tvew
    --环境大家安装官网的教程下载torch，我的版本是torch1.7.1 大家安装自己的cuda版本安装对应的torch就行。
    --然后其他库的版本可以参考requirement.txt
    
## 3.运行
    --可以直接修改main内的参数配置，后直接运行main.py。
    --摄像头 python main.py --cfg_file lib/hand_lib/cfg/handpose.cfg --is_video True --video_path 0
    --视频 python main.py --cfg_file lib/hand_lib/cfg/handpose.cfg --is_video True --video_path test.mp4
    --图片 python main.py --cfg_file lib/hand_lib/cfg/handpose.cfg --is_video False --test_path test_gesture/

## 4.后记
    --大家有兴趣可以直接去看EricLee作者写的组件，真滴不戳。
    --我这边是由于太多人找我问这方面问题，这阵子也很忙没办法一个一个回复。所以把代码整理了一下上传上去。
    --大家有什么建议也可以联系我，我有时间会看。
    
## 6.TODO
    --把动态手势和静态手势完善好吧（如果有时间的话）