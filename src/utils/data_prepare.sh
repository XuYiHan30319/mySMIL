#!/bin/bash

# 指定文件夹路径
input_dir="../../data/PKG_UPENN-GBM_v2/NDPI_images"
base_output_dir="./../data/PKG_UPENN-GBM_v2/NDPI_images_processed"
tile_size=360
n=1024
out_size=224

# 检查输入目录是否存在
if [ ! -d "$input_dir" ]; then
  echo "输入目录不存在：$input_dir"
  exit 1
fi

# 检查基础输出目录是否存在，不存在则创建
if [ ! -d "$base_output_dir" ]; then
  mkdir -p "$base_output_dir"
fi

# 遍历输入目录下的所有文件
for file in "$input_dir"/*; do
  # 获取文件名（不包含扩展名）
  filename=$(basename "$file")
  filename_without_ext="${filename%.*}"
  
  # 创建特定文件的输出目录
  output_dir="$base_output_dir/$filename_without_ext"
  if [ ! -d "$output_dir" ]; then
    mkdir -p "$output_dir"
  fi
  
  # 运行Python脚本
  python sample_tiles.py \
  --input_slide "$input_dir/$filename" \
  --output_dir "$output_dir" \
  --tile_size "$tile_size" \
  --n "$n" \
  --out_size "$out_size"
done

echo "所有文件处理完毕！"
