#!/usr/bin/env python3
from pdf_craft import PDFPageExtractor, MarkDownWriter
import os
import glob
import time
import sys
import shutil
import argparse

def main():
    parser = argparse.ArgumentParser(description="使用本地模型将PDF转换为Markdown")
    parser.add_argument("--model-dir", type=str, default="./models",
                        help="本地模型目录，默认为 './models'")
    parser.add_argument("--output-dir", type=str, default="./output",
                        help="输出目录，默认为 './output'")
    parser.add_argument("pdf_file", type=str, nargs="?", default=None,
                        help="要处理的PDF文件名或路径，如不指定则处理当前目录下所有PDF文件")

    args = parser.parse_args()

    # 设置本地模型目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    local_model_dir = os.path.abspath(args.model_dir)

    # 检查本地模型文件夹
    yolo_model_path = os.path.join(local_model_dir, "yolo", "doclayout_yolo_ft.pt")
    if not os.path.exists(yolo_model_path) or os.path.getsize(yolo_model_path) == 0:
        print("错误：本地YOLO模型文件不存在或为空")
        print(f"请先运行 python download_models.py --dir {args.model_dir} 下载模型文件")
        print("或者使用VPN连接手动从以下地址下载:")
        print("  https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/resolve/main/models/Layout/YOLO/doclayout_yolo_ft.pt")
        print("并保存到以下路径:")
        print(f"  {yolo_model_path}")
        return 1

    # 检查系统模型目录并在需要时复制模型文件
    pdf_craft_models_dir = os.path.expanduser("~/pdf_craft_models")
    system_yolo_model_path = os.path.join(pdf_craft_models_dir, "yolo", "doclayout_yolo_ft.pt")
    
    if (not os.path.exists(system_yolo_model_path) or 
        os.path.getsize(system_yolo_model_path) == 0):
        print(f"系统模型目录中的YOLO模型不存在或为空，正在从本地模型目录复制...")
        os.makedirs(os.path.join(pdf_craft_models_dir, "yolo"), exist_ok=True)
        try:
            shutil.copy2(yolo_model_path, system_yolo_model_path)
            print(f"已复制YOLO模型到系统目录: {system_yolo_model_path}")
        except Exception as e:
            print(f"复制模型文件时出错: {str(e)}")
            return 1
    
    # 复制其他OCR模型
    ocr_models = [
        ("onnx_ocr/ppocrv4/rec", "rec.onnx"),
        ("onnx_ocr/ppocrv4/cls", "cls.onnx"),
        ("onnx_ocr/ppocrv4/det", "det.onnx"),
        ("onnx_ocr/ch_ppocr_server_v2.0", "ppocr_keys_v1.txt")
    ]
    
    for model_path, model_file in ocr_models:
        local_path = os.path.join(local_model_dir, model_path, model_file)
        system_path = os.path.join(pdf_craft_models_dir, model_path, model_file)
        
        if os.path.exists(local_path) and (not os.path.exists(system_path) or 
                                           os.path.getsize(system_path) == 0):
            os.makedirs(os.path.join(pdf_craft_models_dir, model_path), exist_ok=True)
            try:
                shutil.copy2(local_path, system_path)
                print(f"已复制模型 {model_file} 到系统目录")
            except Exception as e:
                print(f"复制模型文件 {model_file} 时出错: {str(e)}")

    # 设置输出目录
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 获取要处理的PDF文件
    if args.pdf_file:
        if os.path.exists(args.pdf_file) and args.pdf_file.endswith('.pdf'):
            pdf_files = [args.pdf_file]
        else:
            pdf_files = glob.glob(os.path.join(current_dir, "*.pdf"))
            pdf_files = [f for f in pdf_files if os.path.basename(f).lower().startswith(args.pdf_file.lower())]
            if not pdf_files:
                print(f"错误: 未找到匹配的PDF文件 '{args.pdf_file}'")
                return 1
    else:
        pdf_files = glob.glob(os.path.join(current_dir, "*.pdf"))
        if not pdf_files:
            print("错误：当前目录下未找到PDF文件")
            return 1

    # 初始化PDF页面提取器
    print("初始化PDF页面提取器...")
    max_retries = 3
    retry_delay = 5  # 秒

    for attempt in range(max_retries):
        try:
            extractor = PDFPageExtractor(
                device="cpu",  # 使用CPU处理
                model_dir_path=pdf_craft_models_dir  # 使用系统模型路径
            )
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"初始化失败，尝试重试 ({attempt+1}/{max_retries})...")
                print(f"错误: {str(e)}")
                time.sleep(retry_delay)
            else:
                print(f"初始化失败，已达到最大重试次数: {str(e)}")
                return 1

    # 处理PDF文件
    for pdf_path in pdf_files:
        pdf_filename = os.path.basename(pdf_path)
        pdf_name = os.path.splitext(pdf_filename)[0]
        
        # 设置输出Markdown文件路径
        markdown_dir = os.path.join(output_dir, pdf_name)
        os.makedirs(markdown_dir, exist_ok=True)
        markdown_path = os.path.join(markdown_dir, f"{pdf_name}.md")
        
        print(f"\n开始处理: {pdf_filename}")
        print(f"输入文件: {pdf_path}")
        print(f"输出文件: {markdown_path}")
        
        # 开始转换过程，增加重试机制
        for attempt in range(max_retries):
            try:
                with MarkDownWriter(markdown_path, "images", "utf-8") as md:
                    for block in extractor.extract(pdf=pdf_path):
                        md.write(block)
                print(f"✓ 转换完成: {pdf_filename} -> {markdown_path}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"转换失败，尝试重试 ({attempt+1}/{max_retries})...")
                    print(f"错误: {str(e)}")
                    time.sleep(retry_delay)
                else:
                    print(f"✗ 转换失败: {pdf_filename}")
                    print(f"  错误信息: {str(e)}")

    print("\n全部处理完成!")
    print(f"转换结果保存在: {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 