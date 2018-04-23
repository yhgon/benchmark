# benchmark for multiple framework 
this repository will share how to benchmark some DL training and inferencing. 

# image
-  imagenet benchmark with pytorch 
--  prepare dataset
downloadd the dataset from Imagenet as below 

```
-rw-rw-r--    1 hryu hryu  13G  4월 23 11:18 ILSVRC2012_img_test.tar
-rw-rw-r--    1 hryu hryu 138G  4월 23 10:45 ILSVRC2012_img_train.tar
-rw-rw-r--    1 hryu hryu 6.3G  4월 23 11:00 ILSVRC2012_img_val.tar

```

--  prepare docker

pull official NGC pytorch containers from NGC repositories. 
```
docker login nvcr.io
docker pull nvcr.io/nvidia/pytorch:18.03-py3
```

for nvidia-docker 2.0, you could run GPU enabaled docker 

```
docker run --runtime=nvidia --shm-size=1g --ulimit memlock=-1 -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  --rm  -v /raid/hryu/datasets/imagenet:/imagenet --name=hryu-pt-basic -ti nvcr.io/nvidia/pytorch:18.03-py3
```

for nvidia-docker 1.0 use below script 
```
nvidia-docker run --shm-size=1g --ulimit memlock=-1 --rm  -v /raid/hryu/datasets/imagenet:/imagenet --name=hryu-pt-basic -ti nvcr.io/nvidia/pytorch:18.03-py3
```

--  run benchmark code
NVIDIA's official pytorch containers already include simple imagenet benchmark code. 

```
cd examples/imagenet/
python -m multiproc ./main.py -a resnet152  --epochs 1 -b 128  --lr 0.01 /imagenet
```

-  horovod benchmark with tensorflow
--  custom dockerbuild
--  horovod configuration
--  run benchmark code
     
# inference with trt
- googlenet with fp16(tensorcore)
-- prepare NGC

```
docker login nvcr.io
docker pull nvcr.io/nvidia/tensorflow:18.03-py2

```

for nvidia-docker 2.0 
```
docker run --runtime=nvidia   --rm  -ti --name=hryu-trt nvcr.io/nvidia/tensorrt:18.03-py2
```
for nvidia-docker 1.0 you could launch docker 
```
nvidia-docker run   --rm  -ti --name=hryu-trt nvcr.io/nvidia/tensorrt:18.03-py2
```

check GPU is work well on docker or not with nvidia system management interface tool(nvidia-smi)
```
root@460314cbdcb6:/workspace/tensorrt/bin# nvidia-smi
Mon Apr 23 09:01:07 2018       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 384.125                Driver Version: 384.125                   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-DGXS...  On   | 00000000:08:00.0 Off |                    0 |
| N/A   40C    P0    38W / 300W |     10MiB / 16149MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

```
within docker,  check the work directory /workspace/tensorrt/samples and check sample code giexec

```
root@460314cbdcb6:/workspace# ls   
README.md  tensorrt  tensorrt_server
root@460314cbdcb6:/workspace# cd /workspace/tensorrt/samples/giexec
root@460314cbdcb6:/workspace/tensorrt/samples/giexec# ls
Makefile  giexec.cpp

```
compile the sample code giexec with make then binary would be located in /worksapce/tensorrt/bin folder
```
root@460314cbdcb6:/workspace/tensorrt/samples/giexec# make
../Makefile.config:6: CUDA_INSTALL_DIR variable is not specified, using /usr/local/cuda-9.0 by default, use CUDA_INSTALL_DIR=<cuda_directory> to change.
../Makefile.config:9: CUDNN_INSTALL_DIR variable is not specified, using  by default, use CUDNN_INSTALL_DIR=<cudnn_directory> to change.
Compiling: giexec.cpp
Linking: ../../bin/giexec_debug
Compiling: giexec.cpp
root@460314cbdcb6:/workspace/tensorrt/samples/giexec# cd ../../bin/
root@460314cbdcb6:/workspace/tensorrt/bin# ls
chobj  dchobj  download-digits-model.py  giexec  giexec_debug

```
for check standard model in caffe, you could download some standard  prototxt script from model zoo as below. 

```
root@460314cbdcb6:/workspace/tensorrt/samples/giexec#  wget https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt
mv deploy.prototxt googlenet.prototxt
root@460314cbdcb6:/workspace/tensorrt/samples/giexec#  wget https://raw.githubusercontent.com/KaimingHe/deep-residual-networks/master/prototxt/ResNet-101-deploy.prototxt
```



-- run benchmark code

check the argument option  --batch, --int8 -- half2 --deploy option. 
```
root@460314cbdcb6:/workspace/tensorrt/bin# giexec 

Mandatory params:
  --deploy=<file>      Caffe deploy file
  --output=<name>      Output blob name (can be specified multiple times)

Optional params:
  --model=<file>       Caffe model file (default = no model, random weights used)
  --batch=N            Set batch size (default = 1)
  --device=N           Set cuda device to N (default = 0)
  --iterations=N       Run N iterations (default = 10)
  --avgRuns=N          Set avgRuns to N - perf is measured as an average of avgRuns (default=10)
  --workspace=N        Set workspace size in megabytes (default = 16)
  --half2              Run in paired fp16 mode (default = false)
  --int8               Run in int8 mode (default = false)
  --verbose            Use verbose logging (default = false)
  --hostTime         Measure host time rather than GPU time (default = false)
  --engine=<file>      Generate a serialized GIE engine
  --calib=<file>       Read INT8 calibration cache file
```

check the multiple configuration with multiple case of batch size, precision(int8, fp16, fp32) and models(googlenet,resnet)
```
root@460314cbdcb6:/workspace/tensorrt/bin# ./giexec --deploy=./googlenet.prototxt --output=prob  --batch=1
root@460314cbdcb6:/workspace/tensorrt/bin# ./giexec --deploy=./googlenet.prototxt --output=prob --half2 --batch=1
root@460314cbdcb6:/workspace/tensorrt/bin# ./giexec --deploy=./googlenet.prototxt --output=prob --int8 --batch=1

root@460314cbdcb6:/workspace/tensorrt/bin# ./giexec --deploy=./ResNet-101-deploy.prototxt --output=prob  --batch=1
root@460314cbdcb6:/workspace/tensorrt/bin# ./giexec --deploy=./ResNet-101-deploy.prototxt --output=prob --half2 --batch=1
root@460314cbdcb6:/workspace/tensorrt/bin# ./giexec --deploy=./ResNet-101-deploy.prototxt --output=prob --int8 --batch=1

root@460314cbdcb6:/workspace/tensorrt/bin# ./giexec --deploy=./googlenet.prototxt --output=prob  --batch=4
root@460314cbdcb6:/workspace/tensorrt/bin# ./giexec --deploy=./googlenet.prototxt --output=prob --half2 --batch=4
root@460314cbdcb6:/workspace/tensorrt/bin# ./giexec --deploy=./googlenet.prototxt --output=prob --int8 --batch=4

root@460314cbdcb6:/workspace/tensorrt/bin# ./giexec --deploy=./ResNet-101-deploy.prototxt --output=prob  --batch=4
root@460314cbdcb6:/workspace/tensorrt/bin# ./giexec --deploy=./ResNet-101-deploy.prototxt --output=prob --half2 --batch=4
root@460314cbdcb6:/workspace/tensorrt/bin# ./giexec --deploy=./ResNet-101-deploy.prototxt --output=prob --int8 --batch=4
```

below is table for formamance 
```
root@460314cbdcb6:/workspace/tensorrt/bin# ./giexec --deploy=./googlenet.prototxt --output=prob --half2 --batch=128
deploy: ./googlenet.prototxt
output: prob
half2
batch: 128
Input "data": 3x224x224
Output "prob": 1000x1x1
name=data, bindingIndex=0, buffers.size()=2
name=prob, bindingIndex=1, buffers.size()=2
Average over 10 runs is 12.8707 ms.
Average over 10 runs is 12.9181 ms.
Average over 10 runs is 12.9096 ms.
Average over 10 runs is 12.8568 ms.
Average over 10 runs is 12.8694 ms.
Average over 10 runs is 12.8638 ms.
Average over 10 runs is 12.8555 ms.
Average over 10 runs is 12.8636 ms.
Average over 10 runs is 12.8473 ms.
Average over 10 runs is 12.8554 ms.

```

