# TinyInfiniTensor

一个简化版的 ai compiler，用于初学者快速上手学习，保留了计算图和 kernel 层的概念，能够基于 c++ 搭建计算图进行推理计算，目前只支持 cpu 平台。

[环境部署文档](docs/项目部署.md)

[训练营作业介绍文档](docs/训练营作业介绍.md)

[文档导航与阅读顺序](docs/文档导航与阅读顺序.md)

[学习前置清单](docs/学习前置清单.md)

[项目现状与能力边界（2026-02-22）](docs/项目现状与能力边界.md)

## 构建说明（默认 clang-14）

项目的 `Makefile` 已默认使用 `clang-14/clang++-14` 构建，直接执行：

```bash
make build
```

默认构建目录为 `build/clang14-Release`，后续可直接复用，不需要重新配置编译器。

执行测试：

```bash
make test-cpp
```

如需切换编译器，可覆盖变量：

```bash
make build C_COMPILER=gcc-10 CXX_COMPILER=g++-10
```
