# 浏览器界面源码

这里包含人体动作、地形与历史控制实验的网页实现。各页面由相应Python服务提供模型状态、骨架和场景数据。

主要服务入口从 [scripts索引](../scripts/README.md) 查看。源码不附带历史模型和地图制品，页面的实际运行配置应绑定本机准备的资源。

项目中已有WASD、Xbox双摇杆、地图选择与同窗口调参实现；不同实验页面的控制与模型版本通过运行记录区分。

[terrain_fit_review](terrain_fit_review/index.html) 是离线查看页：播放BONES源动作与拟合的地面、支撑箱体，数据由 [fit_bones_terrain_support.py](../scripts/fit_bones_terrain_support.py) 导出；从仓库根目录启动静态服务后打开，地址参数 `?index=` 指向导出的 `viewer/index.json`。
