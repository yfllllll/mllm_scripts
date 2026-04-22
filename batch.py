# -*- coding: utf-8 -*-
import os
import argparse
from openai import OpenAI
import json
import ast
import base64
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import time

# 颜色定义
additional_colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple',
                     'brown', 'gray', 'beige', 'turquoise', 'cyan', 'magenta',
                     'lime', 'navy', 'maroon', 'teal', 'olive', 'coral',
                     'lavender', 'violet', 'gold', 'silver']

# 固定检测类别（不再作为命令行参数）
TARGET_CATEGORIES = ["夜间火焰", "夜间烟雾", "夜间烟囱烟雾"]
# ====================== 辅助函数 ======================
def pil_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def call_vllm_chat(messages: list, temperature: float = 0, max_tokens: int = 9280) -> str:
    client = OpenAI(
        base_url="http://localhost:9007/v1",
        api_key="EMPTY"
    )
    try:
        response = client.chat.completions.create(
            model="QwenVl",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"vLLM API 调用失败: {e}")

def parse_json(json_output):
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output

def plot_bounding_boxes(im, bounding_boxes_str, save_path=None):
    img = im.copy()
    width, height = img.size
    draw = ImageDraw.Draw(img)
    colors = additional_colors

    bounding_boxes_str = parse_json(bounding_boxes_str)

    try:
        font = ImageFont.truetype("./wqy-microhei.ttc", size=14)
    except:
        font = ImageFont.load_default()

    try:
        json_output = ast.literal_eval(bounding_boxes_str)
    except:
        try:
            json_output = json.loads(bounding_boxes_str)
        except:
            try:
                end_idx = bounding_boxes_str.rfind('"}') + len('"}')
                truncated_text = bounding_boxes_str[:end_idx] + "]"
                json_output = ast.literal_eval(truncated_text)
            except:
                print("无法解析边界框 JSON")
                return img

    if not isinstance(json_output, list):
        json_output = [json_output]

    for i, bounding_box in enumerate(json_output):
        color = colors[i % len(colors)]
        if "bbox_2d" not in bounding_box:
            continue

        abs_y1 = int(bounding_box["bbox_2d"][1] / 1000 * height)
        abs_x1 = int(bounding_box["bbox_2d"][0] / 1000 * width)
        abs_y2 = int(bounding_box["bbox_2d"][3] / 1000 * height)
        abs_x2 = int(bounding_box["bbox_2d"][2] / 1000 * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1
        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y1

        draw.rectangle(((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=3)

        if "label" in bounding_box:
            draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)

    if save_path:
        img.save(save_path)
    return img

def process_single_image(image_path, output_dir, categories, api_base="http://localhost:9007/v1"):
    try:
        image = Image.open(image_path).convert("RGB")
        img_base64 = pil_to_base64(image)
        img_url = f"data:image/jpeg;base64,{img_base64}"

        categories_str = "、".join(categories)
        prompt = f"""检测图像中的{categories_str}。以 JSON 格式报告边界框坐标"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        response_text = call_vllm_chat(messages, temperature=0, max_tokens=4096)

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_image_path = os.path.join(output_dir, f"{base_name}_detected.jpg")
        out_text_path = os.path.join(output_dir, f"{base_name}_response.txt")

        plot_bounding_boxes(image, response_text, save_path=out_image_path)

        with open(out_text_path, "w", encoding="utf-8") as f:
            f.write(response_text)

        print(f"✓ 处理成功: {image_path} -> {out_image_path}")
        return True
    except Exception as e:
        print(f"✗ 处理失败: {image_path}, 错误: {e}")
        return False

def batch_process_images(input_dir, output_dir, categories, api_base="http://localhost:9007/v1", delay=1):
    os.makedirs(output_dir, exist_ok=True)

    supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    image_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(supported_exts):
                image_files.append(os.path.join(root, file))

    total = len(image_files)
    print(f"找到 {total} 张图像，开始处理...")

    success_count = 0
    for idx, img_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{total}] 处理: {img_path}")
        if process_single_image(img_path, output_dir, categories, api_base):
            success_count += 1
        time.sleep(delay)

    print(f"\n处理完成！成功: {success_count}/{total}")

def main():
    parser = argparse.ArgumentParser(description="批量目标检测脚本（基于 vLLM API）")
    parser.add_argument("--input_dir", type=str, default='/mnt/disk/liangqh/Data/烟火数据',
                        help="输入图像文件夹路径")
    parser.add_argument("--output_dir", type=str, default='/mnt/disk/liangqh/Data/烟火数据_res',
                        help="输出结果文件夹路径")
    parser.add_argument("--api_base", type=str, default="http://localhost:9007/v1",
                        help="vLLM API 地址")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="每张图像处理后的等待时间（秒）")

    args = parser.parse_args()

    batch_process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        categories=TARGET_CATEGORIES,   # 固定使用脚本内定义的类别
        api_base=args.api_base,
        delay=args.delay
    )

if __name__ == "__main__":
    main()