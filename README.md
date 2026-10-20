# xtop

80 列终端里的资源监控。只读，不交互。标准库实现，无第三方依赖。

## 安装

```sh
pip install dist/xtop-0.1.0-py3-none-any.whl
xtop
```

开发时：

```sh
pip install -e .
```

## 用法

```sh
xtop            # 每 2 秒采样
xtop -d 2       # 采样间隔，秒
```

`q` 退出。

## 显示

顶栏是主机、运行时间、load、任务数。下面每类设备一行：

- CPU 占用，内存占用和用量
- 每张 GPU 一行：占用、显存、温度、功耗
- 有 IPv4 地址的网卡：上下行带宽
- 物理磁盘：挂载点、读写带宽、空闲、总量

其余空间是进程表，包含 root。列为 PID、USER、CPU、MEM、RSS、CMD。数字右对齐，单位紧贴数字；终端变窄时只截断命令和挂载点。

## 数据来源

`/proc` 读 CPU、内存、磁盘、网络和进程。GPU 调 `nvidia-smi`，没有则跳过。磁盘只统计有硬件节点的整盘，挂载点通过 sysfs 的 device-mapper 名字还原。
