#!/bin/bash

# 定义要处理的分支列表
BRANCHES=(
  "AaronBoat-dev"
  "deepsource-autofix-84337cbb"
  "deepsource-autofix-d44bb31a"
  "dev"
  "exp"
  "local"
  "main"
  "nvidia"
  "revert-12-dev"
  "stable"
  "submit"
)

# 获取最新远程分支信息
git fetch origin

# 循环处理每个分支
for branch in "${BRANCHES[@]}"
do
  echo "==================== Processing branch: $branch ===================="

  # 创建本地分支并追踪远程分支
  git checkout -b "$branch" "origin/$branch" || { echo "Failed to checkout $branch"; continue; }

  # 删除 .gitattributes 文件（如果存在）
  if [ -f ".gitattributes" ]; then
    rm .gitattributes
    git rm --cached .gitattributes
    echo "Removed .gitattributes from $branch"
  else
    echo ".gitattributes not found in $branch"
  fi

  # 检查是否包含 LFS 文件（例如 tools/Deep3D 是 submodule 或 LFS 文件）
  if git ls-files | grep -q 'tools/Deep3D'; then
    git rm --cached tools/Deep3D
    echo "Removed tools/Deep3D from LFS tracking in $branch"
  fi

  # 添加其他可能的 LFS 文件（可选，按需添加）
  # git rm --cached large_file.bin

  # 提交更改
  git add .
  git commit -m "removed LFS tracking"

  # 推送更改到远程分支（使用 force 可选，仅当需要重写历史时）
  git push origin "$branch"

  echo "✅ Finished processing branch: $branch"
done

echo "All branches processed."