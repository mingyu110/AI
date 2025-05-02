#!/usr/bin/env python3
import os
import sys
import requests
import argparse
from pathlib import Path
from tqdm import tqdm

# 模型下载链接
MODEL_URLS = {
    # YOLO 模型
    "yolo": {
        "doclayout_yolo_ft.pt": "https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/resolve/main/models/Layout/YOLO/doclayout_yolo_ft.pt"
    },
    # OCR 模型
    "onnx_ocr/ppocrv4/rec": {
        "rec.onnx": "https://huggingface.co/moskize/OnnxOCR/resolve/main/ppocrv4/rec/rec.onnx"
    },
    "onnx_ocr/ppocrv4/cls": {
        "cls.onnx": "https://huggingface.co/moskize/OnnxOCR/resolve/main/ppocrv4/cls/cls.onnx"
    },
    "onnx_ocr/ppocrv4/det": {
        "det.onnx": "https://huggingface.co/moskize/OnnxOCR/resolve/main/ppocrv4/det/det.onnx"
    },
    "onnx_ocr/ch_ppocr_server_v2.0": {
        "ppocr_keys_v1.txt": "https://huggingface.co/moskize/OnnxOCR/resolve/main/ch_ppocr_server_v2.0/ppocr_keys_v1.txt"
    }
}

def download_file(url, dest_path, chunk_size=8192):
    """
    下载文件并显示进度条
    """
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(dest_path, 'wb') as f, tqdm(
                desc=dest_path.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"下载 {url} 失败: {str(e)}")
        if dest_path.exists():
            dest_path.unlink()
        return False

def download_models(base_dir, use_modelscope=False):
    """
    下载所有模型到指定目录
    """
    base_dir = Path(base_dir)
    
    # ModelScope 可能需要不同的下载链接，如果有需要可以切换
    if use_modelscope:
        print("ModelScope 下载暂未实现，将使用 Hugging Face 下载")
    
    print(f"将模型下载到 {base_dir} 目录")
    
    # 创建模型目录
    base_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    
    for model_dir, files in MODEL_URLS.items():
        for file_name, url in files.items():
            dest_path = base_dir / model_dir / file_name
            print(f"下载 {file_name} 到 {dest_path}...")
            
            if dest_path.exists() and dest_path.stat().st_size > 0:
                print(f"文件已存在，跳过: {dest_path}")
                success_count += 1
                continue
            
            if download_file(url, dest_path):
                success_count += 1
                print(f"成功下载 {file_name}")
            else:
                fail_count += 1
    
    print(f"\n下载完成，成功: {success_count}，失败: {fail_count}")
    
    # 复制模型到系统目录
    system_dir = Path.home() / "pdf_craft_models"
    copy_to_system = input(f"\n是否将模型复制到系统目录 {system_dir}? (y/n): ").strip().lower()
    
    if copy_to_system == 'y':
        import shutil
        system_dir.mkdir(parents=True, exist_ok=True)
        
        for model_dir, files in MODEL_URLS.items():
            for file_name in files.keys():
                src_path = base_dir / model_dir / file_name
                dest_path = system_dir / model_dir / file_name
                
                if src_path.exists() and src_path.stat().st_size > 0:
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dest_path)
                    print(f"已复制 {src_path} 到 {dest_path}")
        
        print(f"\n文件已复制到系统目录 {system_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载 pdf-craft 所需的模型文件")
    parser.add_argument("--dir", type=str, default="./models", 
                       help="保存模型的目录，默认为 './models'")
    parser.add_argument("--use-modelscope", action="store_true", 
                       help="是否使用 ModelScope 下载（默认使用 Hugging Face）")
    
    args = parser.parse_args()
    download_models(args.dir, args.use_modelscope) 