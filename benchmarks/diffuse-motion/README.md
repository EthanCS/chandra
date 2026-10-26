# 漫反射稳定性与自由相机收尾（2026-10-26）

用户在实机操作后确认当前 indirect diffuse 效果满意，要求停止性能测量并收尾提交。随后报告的“自由相机无法旋转”也已修复，并通过实际右键、F 看向及状态日志核对。本次收尾依据为用户效果确认、31 项 Rust CPU/GPU 回归、2 项 Python 检测器回归和实际交互记录；不把历史性能数据写成最终版本的性能结论。

## 修正内容

- 连续移动局部灯按光照运动处理，不再每帧当作强度突变进行 25 ms 的历史冲刷；真正强度/颜色编辑继续使用独立排能窗口。
- 自适应补采样保持相同采样前缀，供次级传输使用的首条命中不随预算改变。每条 primary ray 的 deposit 权重除以该像素的样本数，防止 32/64 spp 的边界像素压过普通像素、随视角变化改变缓存中的能量比例。
- 默认稳定表面 2 spp，改光/几何运动 4 spp，普通/快速相机运动 8/16 spp；孤立边界使用 16/32 spp，缺少可靠历史的表面以 64 spp 预热四帧。显式固定样本数可用于诊断。独立队列参考路径使用固定预算，以保持其内存上限。
- 在 albedo、直接光和曝光合成前重建漫反射 irradiance。历史验证使用物体编号、法线、世界位置、平面距离及双线性覆盖；新露出的表面不继承旧屏幕颜色。空间滤波与前后帧历史读写分离，历史限制在当前兼容表面的颜色范围内，真正的能量编辑不会被长时间积累挡住。
- 增加转光、局部灯、阶跃、摆动、相机平移/旋转/组合/停止/移出再返回、120 Hz 及全动态组合诊断。自动采集不接收真实键鼠输入，并核对相机/动画元数据，避免用户键入改变某一次参考机位。

## 自由相机

原问题在于 `run_ui` 的全屏根界面也会被 egui 视为可悬停区域。旧判断把透明视口一起当作控制面板，拦住右键拖动；控件键盘焦点还会挡住 F 看向。

现在按实际面板响应矩形判断视口输入。右键从视口开始后可持续捕获；F 可主动释放控件焦点并进入看向。开始手动看向立即停止 orbit，失焦、松开右键、退出 F 模式或关闭 worker 时释放鼠标。Windows 使用实际光标重置落点，避免窗口部分移出屏幕时出现静止漂移。短按方向键也不会因 down/up 都发生在两帧间而丢失。

- 按住右键并移动鼠标，或按 F 后移动鼠标：旋转；再次按 F：释放。
- WASD：移动；Q/E、Space：上下；左右 Shift/Ctrl：快/慢速。
- R：恢复默认相机；Tab/F1：面板显隐；隐藏时也可点击恢复按钮。
- 实机日志记录了 F 模式与真实右键拖动的 yaw/pitch 变化、停止自动环绕、静止不漂移及 Tab 双向切换。精简结果及最终 EXE SHA-256 见 [closeout.json](closeout.json)。完整操作日志留在 `.local/diffuse-motion/`。

## 图像实验与最终验收的区别

[experiments.json](experiments.json) 保留各轮 EXE/分析器 hash、成功项和失败项。原始逐帧文件、manifest、相机、位置、法线及 GIF 位于 `.local/diffuse-motion/`。所有对照均按相同模拟帧和曝光，比较同一 RC 混合渲染器的 64 primary spp 原始输出；这不是路径追踪真值。最终相机 UI 修正后没有重新跑完整图像/性能矩阵，不能把较早二进制的结果标成最终版本的全面量化放行。

检测器保留屏幕空间及点参考的原始事件计数。对原始 RC 参考在边角出现的稀疏暗孔，还报告一个像素范围内、同物体/同法线/同平面的参考支持；跨物体、法线和并行平面不会借用支持。旧版本的数千个局部事件仍会被检出；不能以较小全屏 MAE 或这类参考孔洞的去除来声称任意轨迹在数学上绝对零误差。检测器的投影、接收面隔离另有独立 CPU 测试。

## 复现入口

```powershell
./run-lighting-lab.ps1 -Test
python tools/test-diffuse-motion.py
./run-lighting-lab.ps1 -Courtyard
# 可选重新采集；本次按用户要求停止继续性能测量。
python tools/validate-diffuse-motion.py --capture --output .local/diffuse-motion/new-candidate
```

局部诊断可使用 `--quality-check --quality-camera-track return --quality-frames 224 --quality-tail 6 --quality-surfaces --diffuse-diagnostics`。`diffuse-raw`、`diffuse-spatial`、`diffuse-temporal` 是未曝光 irradiance；最后一项的第四通道为历史年龄，不能当作图像 surface mask。`diffuse-samples-*.u32` 保存实际像素预算。

间接高光、GGX 和 ReSTIR 仍未接入。PICA/机器人及外部资产导入不在本次收尾范围；[第一批验证目录](../diffuse/README.md)保留基础 RC、薄墙、容量恢复和历史测量。
