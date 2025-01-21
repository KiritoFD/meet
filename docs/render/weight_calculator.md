# 骨骼权重计算系统技术文档

## 1. 系统概述

骨骼权重计算系统是一个高性能的3D模型骨骼权重自动计算工具，支持GPU加速和CPU降级方案。系统使用热扩散算法计算顶点权重，并提供了完整的缓存机制。

### 1.1 主要特性

- 支持GPU/CPU自动切换
- 多线程并行计算
- 批量处理优化
- 权重平滑和体积保持
- 高效的缓存机制
- 完整的性能测试套件

## 2. 核心组件

### 2.1 配置类

#### WeightingConfig
控制权重计算的参数配置： 