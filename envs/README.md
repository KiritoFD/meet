# Meet AI Platform 环境配置

## 环境说明

- `dev.yml`: 开发环境配置，包含完整的开发工具
- `prod.yml`: 生产环境配置，只包含运行必需的依赖
- `requirements/`: pip依赖文件目录
  - `dev.txt`: 开发环境pip依赖
  - `prod.txt`: 生产环境pip依赖
  - `tools.txt`: 工具类依赖

## 使用方法

### 开发环境
```bash
conda env create -f envs/dev.yml
```

### 生产环境
```bash
conda env create -f envs/prod.yml
```

### 依赖更新
```bash
conda env update -f envs/dev.yml
```
