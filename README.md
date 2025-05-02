# PDF转Markdown本地模型工具使用说明

本项目使用修改版的脚本，可以使用本地模型文件将PDF转换为Markdown，避免每次运行时都需要从网络下载模型。

## 核心工作流程:

1. 通过download_models.py下载所需模型到本地

2. 运行local_pdf_to_md.py来转换PDF文件

3. 脚本初始化PDFPageExtractor对象并设置为使用CPU(也可以指定使用GPU，速度更快；目前默认使用的是CPU)

4. 遍历指定的PDF文件并使用MarkDownWriter将转换结果写入输出目录

5. 转换过程中包含错误处理和重试机制

## 一、下载模型文件

### 1. 通过提供的脚本下载模型

首先需要下载所需的模型文件。如果您有VPN或网络环境良好，可以使用以下命令下载：

```bash
python download_models.py
```

这将下载以下模型文件到 `./models` 目录：

- YOLO布局检测模型: `models/yolo/doclayout_yolo_ft.pt`
- OCR模型：
  - `models/onnx_ocr/ppocrv4/rec/rec.onnx`
  - `models/onnx_ocr/ppocrv4/cls/cls.onnx`
  - `models/onnx_ocr/ppocrv4/det/det.onnx`
  - `models/onnx_ocr/ch_ppocr_server_v2.0/ppocr_keys_v1.txt`

下载后，模型文件会自动复制到系统目录 `~/pdf_craft_models`，以供转换程序使用。

### 2. 手动下载模型

如果自动下载失败，您可以从以下链接手动下载模型文件：

1. YOLO布局检测模型:
   - https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0/resolve/main/models/Layout/YOLO/doclayout_yolo_ft.pt
   
2. OCR模型:
   - https://huggingface.co/moskize/OnnxOCR/resolve/main/ppocrv4/rec/rec.onnx
   - https://huggingface.co/moskize/OnnxOCR/resolve/main/ppocrv4/cls/cls.onnx
   - https://huggingface.co/moskize/OnnxOCR/resolve/main/ppocrv4/det/det.onnx
   - https://huggingface.co/moskize/OnnxOCR/resolve/main/ch_ppocr_server_v2.0/ppocr_keys_v1.txt

下载后将文件放置到对应的目录结构中。

## 二、使用本地模型转换PDF

下载完模型后，您可以使用本地模型转换PDF文件，建议在Python虚拟环境使用：

```bash
python local_pdf_to_md.py [PDF文件名]
```

### 参数说明

- `--model-dir`: 本地模型目录，默认为 `./models`
- `--output-dir`: 输出目录，默认为 `./output`
- `pdf_file`: 要处理的PDF文件名或路径，如不指定则处理当前目录下所有PDF文件

### 示例

1. 转换特定PDF文件：
   ```bash
   python local_pdf_to_md.py "Design Pattern for MicroService.pdf"
   ```

2. 指定模型目录和输出目录：
   ```bash
   python local_pdf_to_md.py --model-dir /path/to/models --output-dir /path/to/output "Design Pattern for MicroService.pdf"
   ```

3. 转换当前目录下所有PDF文件：
   ```bash
   python local_pdf_to_md.py
   ```

## 三、注意事项

1. 首次运行时，脚本会将本地模型文件复制到系统目录 `~/pdf_craft_models`
2. 如果系统目录中的模型文件已存在且大小正常，将直接使用它们，不会重新复制
3. 转换大型PDF文件可能需要较长时间，请耐心等待
4. 如果转换过程中出现错误，脚本会自动重试最多3次

## 四、模型文件说明

- `doclayout_yolo_ft.pt`: 布局检测模型，用于识别PDF中的文本、表格、图片等元素
- OCR模型文件：用于识别图像中的文字内容
  - `rec.onnx`: 文字识别模型
  - `cls.onnx`: 文本方向分类模型
  - `det.onnx`: 文本检测模型
  - `ppocr_keys_v1.txt`: 字符集文件 