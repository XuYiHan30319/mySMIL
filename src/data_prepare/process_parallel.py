import os
from multiprocessing import Pool
from tqdm import tqdm


def process_file(input_file, base_output_dir, tile_size, n, out_size):
    # 获取文件名（不包含扩展名）
    filename = os.path.basename(input_file)
    filename_without_ext = os.path.splitext(filename)[0]

    # 创建特定文件的输出目录
    output_dir = os.path.join(base_output_dir, filename_without_ext)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 运行Python脚本
    command = f"python sample_tiles.py --input_slide {input_file} --output_dir {output_dir} --tile_size {tile_size} --n {n} --out_size {out_size}"
    os.system(command)


def process_file_wrapper(args):
    return process_file(*args)


if __name__ == "__main__":
    # 指定文件夹路径
    input_dir = "../../data/PKG_CPTAC_LUAD_v12/LUAD"
    base_output_dir = "../../data/PKG_CPTAC_LUAD_v12/LUAD_processed"
    tile_size = 125
    n = 32
    out_size = 224

    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"输入目录不存在：{input_dir}")
        exit(1)

    # 检查基础输出目录是否存在，不存在则创建
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)

    # 获取输入目录下的所有文件
    input_files = [
        os.path.join(input_dir, file)
        for file in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, file))
    ]

    # 并行处理文件
    with Pool() as pool:
        with tqdm(total=len(input_files)) as pbar:
            for _ in pool.imap_unordered(
                process_file_wrapper,
                [
                    (file, base_output_dir, tile_size, n, out_size)
                    for file in input_files
                ],
            ):
                pbar.update()

    print("所有文件处理完毕！")
