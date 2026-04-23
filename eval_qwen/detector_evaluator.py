# -*- coding: utf-8 -*-
"""
检测评估核心模块
支持检测模式（返回边界框）和 VQA 模式（回答是/否）
"""
import os
import time
import json
import ast
import base64
import shutil
import csv
from io import BytesIO
from typing import Dict, List, Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

# 颜色定义
additional_colors = ['red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple',
                     'brown', 'gray', 'beige', 'turquoise', 'cyan', 'magenta',
                     'lime', 'navy', 'maroon', 'teal', 'olive', 'coral',
                     'lavender', 'violet', 'gold', 'silver']


class DetectorEvaluator:
    def __init__(self, api_base: str, model_name: str = "QwenVl", max_tokens: int = 4096):
        self.api_base = api_base
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=api_base, api_key="EMPTY")

    def resize_image_max_side(self, image: Image.Image, max_size: int = 1280) -> Tuple[Image.Image, float, float]:
        """等比缩放图像，返回缩放后的图像和宽高缩放因子"""
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

    def pil_to_base64(self, image: Image.Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def call_api(self, messages: List[Dict]) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"API调用失败: {e}")

    def parse_json(self, text: str) -> str:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip() == "```json":
                return "\n".join(lines[i+1:]).split("```")[0]
        return text

    def has_detection(self, response_text: str) -> bool:
        """检测模式下判断是否包含边界框"""
        try:
            cleaned = self.parse_json(response_text)
            data = ast.literal_eval(cleaned)
            if not isinstance(data, list):
                data = [data]
            for item in data:
                if "bbox_2d" in item:
                    return True
            return False
        except:
            return False

    def parse_vqa_response(self, response_text: str) -> bool:
        """VQA 模式下解析回答，判断是否肯定"""
        text_lower = response_text.lower()
        positive_keywords = ["是", "有", "存在", "yes", "true", "包含", "检测到"]
        negative_keywords = ["否", "没有", "不存在", "no", "false", "未检测到", "无"]
        for word in positive_keywords:
            if word in text_lower:
                return True
        for word in negative_keywords:
            if word in text_lower:
                return False
        return False

    def plot_boxes(self, im: Image.Image, bbox_str: str, save_path: str,
                   scale_w: float = 1.0, scale_h: float = 1.0) -> Image.Image:
        """绘制边界框并保存"""
        img = im.copy()
        width, height = img.size
        draw = ImageDraw.Draw(img)
        colors = additional_colors

        bbox_str = self.parse_json(bbox_str)
        try:
            font = ImageFont.truetype("./wqy-microhei.ttc", size=14)
        except:
            font = ImageFont.load_default()

        try:
            json_output = ast.literal_eval(bbox_str)
        except:
            try:
                json_output = json.loads(bbox_str)
            except:
                try:
                    end_idx = bbox_str.rfind('"}') + len('"}')
                    truncated_text = bbox_str[:end_idx] + "]"
                    json_output = ast.literal_eval(truncated_text)
                except:
                    print("无法解析边界框 JSON")
                    return img

        if not isinstance(json_output, list):
            json_output = [json_output]

        input_w = width / scale_w
        input_h = height / scale_h

        for i, box in enumerate(json_output):
            color = colors[i % len(colors)]
            if "bbox_2d" not in box:
                continue
            x1 = box["bbox_2d"][0] / 1000.0 * input_w * scale_w
            y1 = box["bbox_2d"][1] / 1000.0 * input_h * scale_h
            x2 = box["bbox_2d"][2] / 1000.0 * input_w * scale_w
            y2 = box["bbox_2d"][3] / 1000.0 * input_h * scale_h

            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=3)
            if "label" in box:
                draw.text((x1+8, y1+6), box["label"], fill=color, font=font)

        if save_path:
            img.save(save_path)
        return img

    def evaluate_image(self, image_path: str, is_positive: bool,
                       categories: List[str], prompt_template: str,
                       mode: str = "detection") -> Dict[str, Any]:
        """处理单张图像，返回结果字典"""
        start = time.time()
        result = {
            "image": os.path.basename(image_path),
            "true_label": "positive" if is_positive else "negative",
            "detected": False,
            "category": "",
            "time_sec": 0.0,
            "response": "",
            "error": None,
            "output_image_path": ""
        }
        try:
            original = Image.open(image_path).convert("RGB")
            resized, scale_w, scale_h = self.resize_image_max_side(original, 1280)
            img_b64 = self.pil_to_base64(resized)
            img_url = f"data:image/jpeg;base64,{img_b64}"

            if mode == "detection":
                categories_str = ", ".join(categories)
                prompt = prompt_template.format(categories=categories_str)
            else:  # vqa
                # VQA模式提示词也支持占位符替换
                categories_str = ", ".join(categories)
                prompt = prompt_template.format(categories=categories_str) if "{categories}" in prompt_template else prompt_template

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": img_url}},
                    {"type": "text", "text": prompt}
                ]
            }]

            response = self.call_api(messages)
            elapsed = time.time() - start
            result["time_sec"] = round(elapsed, 3)
            result["response"] = response
            print(f"提示词: {prompt}")
            print(f"处理时间: {result['time_sec']}s")
            print(f"API响应: {response}")

            if mode == "detection":
                detected = self.has_detection(response)
            else:
                detected = self.parse_vqa_response(response)

            result["detected"] = detected

            if is_positive:
                category = "TP" if detected else "FN"
            else:
                category = "FP" if detected else "TN"
            result["category"] = category

            return result

        except Exception as e:
            elapsed = time.time() - start
            result["time_sec"] = round(elapsed, 3)
            result["error"] = str(e)
            return result


def run_evaluation(
    input_dir: str,
    output_dir: str,
    categories: List[str],
    api_base: str,
    model_name: str = "QwenVl",
    prompt_template: str = None,
    mode: str = "detection",
    delay: float = 1.0
) -> Dict[str, Any]:
    """执行一次完整评估，返回统计数据和详细记录"""
    evaluator = DetectorEvaluator(api_base, model_name)

    assert prompt_template is not None, "必须提供 prompt_template 参数"

    # 准备输出目录
    result_folders = {
        "TP": os.path.join(output_dir, "TP"),
        "TN": os.path.join(output_dir, "TN"),
        "FP": os.path.join(output_dir, "FP"),
        "FN": os.path.join(output_dir, "FN")
    }
    for f in result_folders.values():
        os.makedirs(f, exist_ok=True)

    # 收集图像
    pos_dir = os.path.join(input_dir, "positive")
    neg_dir = os.path.join(input_dir, "negative")
    if not os.path.isdir(pos_dir) or not os.path.isdir(neg_dir):
        raise ValueError("输入目录必须包含 'positive' 和 'negative' 子文件夹")

    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
    image_list = []
    pos_count = 0
    neg_count = 0
    for root, _, files in os.walk(pos_dir):
        for f in files:
            if f.lower().endswith(exts):
                image_list.append((os.path.join(root, f), True))
                pos_count += 1
    for root, _, files in os.walk(neg_dir):
        for f in files:
            if f.lower().endswith(exts):
                image_list.append((os.path.join(root, f), False))
                neg_count += 1

    total = len(image_list)
    print(f"数据集: {input_dir} | 模型: {model_name} | 模式: {mode} | 图像总数: {total} (正:{pos_count} 负:{neg_count})")

    tp = tn = fp = fn = 0
    total_time = 0.0
    records = []

    for idx, (img_path, is_pos) in enumerate(image_list, 1):
        print(f"[{idx}/{total}] 处理: {img_path} (正样本: {is_pos})")
        res = evaluator.evaluate_image(img_path, is_pos, categories, prompt_template, mode)

        if res["error"]:
            print(f"  ✗ 处理失败: {res['error']}")
            res["category"] = "ERROR"
            records.append(res)
            continue

        total_time += res["time_sec"]
        cat = res["category"]
        print(f"  ✓ {cat}: {img_path} (耗时 {res['time_sec']:.2f}s)")

        if cat == "TP":
            tp += 1
        elif cat == "TN":
            tn += 1
        elif cat == "FP":
            fp += 1
        elif cat == "FN":
            fn += 1

        # 保存图像
        dest_folder = result_folders[cat]
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        out_name = f"{idx:04d}_{base_name}.jpg"
        out_path = os.path.join(dest_folder, out_name)
        res["output_image_path"] = os.path.relpath(out_path, output_dir)

        if cat in ("TP", "FP"):
            try:
                original = Image.open(img_path).convert("RGB")
                _, sw, sh = evaluator.resize_image_max_side(original, 1280)
                evaluator.plot_boxes(original, res["response"], out_path, sw, sh)
            except Exception as e:
                print(f"  绘制失败: {e}, 复制原图")
                shutil.copy2(img_path, out_path)
        else:
            shutil.copy2(img_path, out_path)

        records.append(res)
        time.sleep(delay)

    # 计算指标
    total_pos = tp + fn
    total_neg = tn + fp
    recall = tp / total_pos if total_pos else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
    miss_rate = fn / total_pos if total_pos else 0
    false_alarm = fp / total_neg if total_neg else 0

    stats = {
        "model": model_name,
        "dataset": os.path.basename(input_dir),
        "mode": mode,
        "prompt": prompt_template,
        "total_images": total,
        "positive_count": pos_count,
        "negative_count": neg_count,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "recall": recall, "precision": precision, "f1": f1,
        "accuracy": accuracy, "miss_rate": miss_rate, "false_alarm": false_alarm,
        "total_time": total_time, "avg_time": total_time/total if total else 0
    }

    # 写入CSV
    detail_csv = os.path.join(output_dir, "details.csv")
    with open(detail_csv, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["image", "true_label", "detected", "category", "time_sec", "output_image_path", "response"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)

    summary_csv = os.path.join(output_dir, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        for k, v in stats.items():
            writer.writerow([k, v])

    print("\n" + "="*50)
    print(f"模型: {model_name}  数据集: {stats['dataset']}  模式: {mode}")
    print(f"正样本: {pos_count}  负样本: {neg_count}")
    print(f"TP:{tp} TN:{tn} FP:{fp} FN:{fn}")
    print(f"查准率:{precision:.4f} 查全率:{recall:.4f} F1:{f1:.4f} 准确率:{accuracy:.4f}")
    print("="*50)

    return stats, records


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--api_base", type=str, default="http://localhost:9007/v1")
    parser.add_argument("--model", type=str, default="QwenVl")
    parser.add_argument("--mode", choices=["detection", "vqa"], default="detection")
    parser.add_argument("--delay", type=float, default=1.0)
    args = parser.parse_args()

    categories = ["夜间火焰", "夜间烟雾", "夜间烟囱烟雾"]
    run_evaluation(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        categories=categories,
        api_base=args.api_base,
        model_name=args.model,
        mode=args.mode,
        delay=args.delay
    )