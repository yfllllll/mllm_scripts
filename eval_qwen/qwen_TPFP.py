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
import shutil
import csv

# 颜色定义
additional_colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple',
                     'brown', 'gray', 'beige', 'turquoise', 'cyan', 'magenta',
                     'lime', 'navy', 'maroon', 'teal', 'olive', 'coral',
                     'lavender', 'violet', 'gold', 'silver']

# 固定检测类别
TARGET_CATEGORIES = ["夜间出现的火焰", "夜间出现的烟雾", "夜间烟囱排放的烟雾"]

# ====================== 辅助函数 ======================
def resize_image_max_side(image: Image.Image, max_size: int = 1280) -> tuple:
    """
    将图像等比缩放，使最大边不超过 max_size。
    返回 (缩放后的图像, 宽度缩放因子, 高度缩放因子)
    """
    orig_w, orig_h = image.size
    if max(orig_w, orig_h) <= max_size:
        return image.copy(), 1.0, 1.0
    if orig_w >= orig_h:
        new_w = max_size
        new_h = int(orig_h * max_size / orig_w)
    else:
        new_h = max_size
        new_w = int(orig_w * max_size / orig_h)
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    scale_w = orig_w / new_w
    scale_h = orig_h / new_h
    return resized, scale_w, scale_h

def pil_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def call_vllm_chat(messages: list, api_base: str, temperature: float = 0, max_tokens: int = 9280) -> str:
    client = OpenAI(
        base_url=api_base,
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

def plot_bounding_boxes(im, bounding_boxes_str, save_path=None, scale_w=1.0, scale_h=1.0):
    """
    在原图上绘制边界框。
    scale_w, scale_h: 从模型输入图像尺寸到原图尺寸的缩放因子（原图宽/输入宽，原图高/输入高）。
    模型返回的坐标是基于千分比（0-1000）的，表示在输入图像上的相对位置，
    需要先转换为输入图像上的绝对坐标，再乘以缩放因子得到原图坐标。
    """
    img = im.copy()
    width, height = img.size   # 原图尺寸
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

    # 模型返回的坐标是基于千分比（0-1000），表示在输入图像（缩放后）上的相对位置
    # 输入图像的宽高 = width / scale_w, height / scale_h
    input_w = width / scale_w
    input_h = height / scale_h

    for i, bounding_box in enumerate(json_output):
        color = colors[i % len(colors)]
        if "bbox_2d" not in bounding_box:
            continue

        # 千分比转输入图像绝对坐标
        x1_input = bounding_box["bbox_2d"][0] / 1000.0 * input_w
        y1_input = bounding_box["bbox_2d"][1] / 1000.0 * input_h
        x2_input = bounding_box["bbox_2d"][2] / 1000.0 * input_w
        y2_input = bounding_box["bbox_2d"][3] / 1000.0 * input_h

        # 映射到原图坐标
        abs_x1 = int(x1_input * scale_w)
        abs_y1 = int(y1_input * scale_h)
        abs_x2 = int(x2_input * scale_w)
        abs_y2 = int(y2_input * scale_h)

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

def has_detection(response_text: str) -> bool:
    try:
        cleaned = parse_json(response_text)
        data = ast.literal_eval(cleaned)
        if not isinstance(data, list):
            data = [data]
        for item in data:
            if "bbox_2d" in item:
                return True
        return False
    except:
        return False

def process_single_image(image_path, output_dir_detected, categories, api_base,
                         is_positive: bool, result_folders: dict):
    start_time = time.time()
    try:
        # 加载原图
        original_image = Image.open(image_path).convert("RGB")
        
        # 缩放图像，同时获得缩放因子
        resized_image, scale_w, scale_h = resize_image_max_side(original_image, max_size=1280)
        img_base64 = pil_to_base64(resized_image)
        img_url = f"data:image/jpeg;base64,{img_base64}"

        categories_str = ",".join(categories)
        prompt = f"""locate every instance that belongs to the following categories: {categories_str}. Report the bounding box coordinates in JSON format. When there are multiple targets, only a maximum of 10 rectangular boxes shall be returned."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        response_text = call_vllm_chat(messages, api_base=api_base, temperature=0, max_tokens=4096)
        elapsed = time.time() - start_time
        print(f"API 响应: {response_text}")  # 打印响应的前200字符以供调试
        detected = has_detection(response_text)

        if is_positive:
            category = "TP" if detected else "FN"
        else:
            category = "FP" if detected else "TN"

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        dest_folder = result_folders[category]

        # 生成保存路径（避免重名）
        dest_path = os.path.join(dest_folder, os.path.basename(image_path))
        if os.path.exists(dest_path):
            name, ext = os.path.splitext(os.path.basename(image_path))
            dest_path = os.path.join(dest_folder, f"{name}_{int(time.time())}{ext}")

        # 修改点：对于 TP 和 FP，保存绘制了检测框的图像；TN 和 FN 保存原图
        if category in ("TP", "FP"):
            # 绘制带框图像并保存到分类文件夹
            plot_bounding_boxes(original_image, response_text, save_path=dest_path,
                                scale_w=scale_w, scale_h=scale_h)
        else:
            # TN 或 FP：直接复制原图
            shutil.copy2(image_path, dest_path)

        # 另外，仍然将带框图像保存到 detected 目录作为备份（可选）
        # if output_dir_detected:
        #     os.makedirs(output_dir_detected, exist_ok=True)
        #     out_image_path = os.path.join(output_dir_detected, f"{base_name}_detected.jpg")
        #     plot_bounding_boxes(original_image, response_text, save_path=out_image_path,
        #                         scale_w=scale_w, scale_h=scale_h)

        print(f"✓ {category}: {image_path} (耗时 {elapsed:.2f}s)")
        return elapsed, detected

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"✗ 处理失败: {image_path}, 错误: {e}")
        return elapsed, False

def batch_process_images(input_dir, output_dir, categories, api_base="http://localhost:9007/v1", delay=1):
    result_folders = {
        "TP": os.path.join(output_dir, "TP"),
        "TN": os.path.join(output_dir, "TN"),
        "FP": os.path.join(output_dir, "FP"),
        "FN": os.path.join(output_dir, "FN")
    }
    for folder in result_folders.values():
        os.makedirs(folder, exist_ok=True)

    detected_dir = os.path.join(output_dir, "detected")
    os.makedirs(detected_dir, exist_ok=True)

    supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    positive_dir = os.path.join(input_dir, "positive")
    negative_dir = os.path.join(input_dir, "negative")

    if not os.path.isdir(positive_dir) or not os.path.isdir(negative_dir):
        print("错误: 输入目录必须包含 'positive' 和 'negative' 两个子文件夹")
        return

    image_list = []
    for root, _, files in os.walk(positive_dir):
        for file in files:
            if file.lower().endswith(supported_exts):
                image_list.append((os.path.join(root, file), True))
    for root, _, files in os.walk(negative_dir):
        for file in files:
            if file.lower().endswith(supported_exts):
                image_list.append((os.path.join(root, file), False))

    total = len(image_list)
    if total == 0:
        print("错误: 未找到任何支持的图像文件")
        return

    total_pos = sum(1 for _, pos in image_list if pos)
    total_neg = total - total_pos
    print(f"找到 {total} 张图像 (正样本: {total_pos}, 负样本: {total_neg})")
    print("开始处理...")

    tp = tn = fp = fn = 0
    total_time = 0.0
    records = []

    for idx, (img_path, is_positive) in enumerate(image_list, 1):
        print(f"\n[{idx}/{total}] 处理: {img_path} (正样本: {is_positive})")
        elapsed, detected = process_single_image(
            image_path=img_path,
            output_dir_detected=detected_dir,
            categories=categories,
            api_base=api_base,
            is_positive=is_positive,
            result_folders=result_folders
        )
        total_time += elapsed

        if is_positive:
            if detected:
                tp += 1
                pred_class = "TP"
            else:
                fn += 1
                pred_class = "FN"
        else:
            if detected:
                fp += 1
                pred_class = "FP"
            else:
                tn += 1
                pred_class = "TN"

        records.append({
            "image": os.path.basename(img_path),
            "true_label": "positive" if is_positive else "negative",
            "detected": detected,
            "result_class": pred_class,
            "time_sec": round(elapsed, 3)
        })
        time.sleep(delay)

    total_pos_actual = tp + fn
    total_neg_actual = tn + fp
    recall = tp / total_pos_actual if total_pos_actual > 0 else 0          # 查全率
    miss_rate = fn / total_pos_actual if total_pos_actual > 0 else 0
    false_alarm_rate = fp / total_neg_actual if total_neg_actual > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0 # 查准率
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

    print("\n" + "="*50)
    print("检测结果统计:")
    print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print(f"漏检率 (FN / Positive): {miss_rate:.4f}")
    print(f"错检率 (FP / Negative): {false_alarm_rate:.4f}")
    print(f"查全率 (Recall TP/ (TP+FN)):    {recall:.4f}")
    print(f"查准率 (TP / (TP+FP)): {precision:.4f}")
    print(f"准确率 ((TP+TN) / Total): {accuracy:.4f}")
    print(f"总推理耗时: {total_time:.2f} 秒, 平均每张: {total_time/total:.3f} 秒")

    print("="*50)

    summary_csv = os.path.join(output_dir, "evaluation_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["指标", "值"])
        writer.writerow(["TP", tp])
        writer.writerow(["TN", tn])
        writer.writerow(["FP", fp])
        writer.writerow(["FN", fn])
        writer.writerow(["漏检率", f"{miss_rate:.4f}"])
        writer.writerow(["错检率", f"{false_alarm_rate:.4f}"])
        writer.writerow(["查全率", f"{recall:.4f}"])
        writer.writerow(["查准率", f"{precision:.4f}"])
        writer.writerow(["准确率", f"{accuracy:.4f}"])
        writer.writerow(["总推理时间(秒)", f"{total_time:.2f}"])
        writer.writerow(["平均推理时间(秒)", f"{total_time/total:.3f}"])

    detail_csv = os.path.join(output_dir, "detection_details.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["image", "true_label", "detected", "result_class", "time_sec"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n结果已保存至 {output_dir}")
    print(f"TP/FP 文件夹内存放带检测框的图像，TN/FN 存放原图。")
    print(f"带框检测图备份存放于: {detected_dir}")
    print(f"汇总指标: {summary_csv}")
    print(f"详细记录: {detail_csv}")

def main():
    parser = argparse.ArgumentParser(description="批量目标检测脚本（基于 vLLM API）")
    parser.add_argument("--input_dir", type=str, default='/mnt/disk/liangqh/Data/烟火数据',
                        help="输入图像文件夹路径，其下应有 positive/ 和 negative/ 子文件夹")
    parser.add_argument("--output_dir", type=str, default='/mnt/disk/liangqh/Data/烟火数据_eva',
                        help="输出结果文件夹路径")
    parser.add_argument("--api_base", type=str, default="http://localhost:9007/v1",
                        help="vLLM API 地址")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="每张图像处理后的等待时间（秒）")

    args = parser.parse_args()
    batch_process_images(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        categories=TARGET_CATEGORIES,
        api_base=args.api_base,
        delay=args.delay
    )

if __name__ == "__main__":
    main()