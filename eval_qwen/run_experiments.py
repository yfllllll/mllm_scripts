# -*- coding: utf-8 -*-
"""
批量实验脚本：支持多任务（数据集）、多模型、多提示词组合
每个任务可以有自己的检测类别
支持追加到已有实验目录
"""
import os
import csv
import argparse
import base64
from datetime import datetime
from detector_evaluator import run_evaluation

# ==================== 默认配置 ====================
DEFAULT_API_BASE = "http://localhost:9007/v1"

# 模型列表
MODELS = [
    {"name": "Qwen3Vl-4B", "api_base": DEFAULT_API_BASE},
    # {"name": "QwenVl-72B", "api_base": "http://another-ip:9007/v1"},
]

# 任务配置：key 为任务名，value 为字典，包含路径和类别列表
TASKS = {
    "烟火检测": {
        "path": "/mnt/disk/liangqh/Data/烟火数据",
        "categories": ["夜间出现的火焰", "夜间出现的烟雾", "夜间烟囱排放的烟雾"]
    },
    "机械施工检测": {
        "path": "/mnt/disk/liangqh/Data/ZGTT-1717104001_机械施工监控",
        "categories": ["挖掘机", "推土机", "渣土车","水泥搅拌车","拖拉机"]
    },
}

# 提示词模板（使用 {categories} 占位符）
PROMPTS = {
    "detection_cn": "检测图像中的以下类别：{categories}。以 JSON 格式报告边界框坐标。",
    #"detection_en": "Locate every instance that belongs to the following categories: {categories}. Report bounding box coordinates in JSON format.",
    "vqa_cn": "请仔细观察图像，判断图中是否存在以下类别：{categories}。只需回答“是”或“否”。",
    #"vqa_en": "Observe the image carefully. Is there any {categories}? Answer only 'yes' or 'no'.",
}

# 默认实验根目录
DEFAULT_OUTPUT_ROOT = "/mnt/disk/liangqh/experiments"
# =================================================


def image_to_base64(img_path):
    """将图片文件转换为 Base64 字符串"""
    try:
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"读取图片失败: {img_path}, {e}")
        return ""


def load_existing_summaries(exp_root):
    """加载已有汇总CSV，返回实验ID集合和记录列表"""
    summary_file = os.path.join(exp_root, "all_experiments_summary.csv")
    existing_ids = set()
    existing_records = []
    if os.path.exists(summary_file):
        with open(summary_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ids.add(row["experiment_id"])
                existing_records.append(row)
        print(f"加载已有实验记录 {len(existing_records)} 条")
    return existing_ids, existing_records


def generate_html_report(exp_root, all_summaries):
    """生成自包含的HTML报告，图片以Base64嵌入"""
    html_path = os.path.join(exp_root, "report.html")

    # 收集每个实验的示例图片路径
    sample_images = {}
    for summary in all_summaries:
        exp_id = summary["experiment_id"]
        exp_dir = os.path.join(exp_root, exp_id)
        samples = {}
        for cat in ["TP", "FP", "TN", "FN"]:
            cat_dir = os.path.join(exp_dir, cat)
            if os.path.isdir(cat_dir):
                files = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
                if files:
                    # 取第一张作为示例
                    rel_path = os.path.join(exp_id, cat, files[0])
                    samples[cat] = rel_path
        sample_images[exp_id] = samples

    # 构建HTML
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>目标检测评估报告</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; background: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: center; }}
        th {{ background-color: #4CAF50; color: white; }}
        .experiment-block {{ background: white; margin-bottom: 40px; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .sample-row {{ display: flex; flex-wrap: wrap; gap: 15px; margin-top: 15px; }}
        .sample-card {{ border: 1px solid #ccc; padding: 10px; border-radius: 5px; background: #fafafa; text-align: center; }}
        .sample-card img {{ max-width: 200px; max-height: 150px; display: block; margin: 0 auto; }}
        .prompt {{ background: #f0f0f0; padding: 10px; border-radius: 4px; font-family: monospace; white-space: pre-wrap; }}
        .stats {{ margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>🔍 目标检测评估报告</h1>
    <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <h2>📊 实验汇总</h2>
    <table>
        <tr>
            <th>实验ID</th><th>模型</th><th>数据集</th><th>任务</th><th>模式</th><th>检测类别</th>
            <th>图像总数</th><th>正样本</th><th>负样本</th>
            <th>TP</th><th>TN</th><th>FP</th><th>FN</th>
            <th>查准率</th><th>查全率</th><th>F1</th><th>准确率</th><th>平均耗时(s)</th>
        </tr>
"""
    for s in all_summaries:
        categories_str = "、".join(s.get('categories', [])) if isinstance(s.get('categories'), list) else s.get('categories', '')
        html_content += f"""
        <tr>
            <td>{s['experiment_id']}</td><td>{s['model']}</td><td>{s['dataset']}</td><td>{s.get('task_name', '')}</td><td>{s['mode']}</td>
            <td>{categories_str}</td>
            <td>{s['total_images']}</td><td>{s['positive_count']}</td><td>{s['negative_count']}</td>
            <td>{s['TP']}</td><td>{s['TN']}</td><td>{s['FP']}</td><td>{s['FN']}</td>
            <td>{s['precision']}</td><td>{s['recall']}</td><td>{s['f1']}</td><td>{s['accuracy']}</td><td>{s['avg_time']}</td>
        </tr>"""
    html_content += "\n    </table>\n"

    # 各实验详情
    html_content += "\n    <h2>📁 各实验详情与示例图片</h2>\n"
    for s in all_summaries:
        exp_id = s["experiment_id"]
        categories_str = "、".join(s.get('categories', [])) if isinstance(s.get('categories'), list) else s.get('categories', '')
        html_content += f"""
    <div class="experiment-block">
        <h3>实验: {exp_id}</h3>
        <p><strong>模型:</strong> {s['model']} | <strong>数据集路径:</strong> {s.get('dataset_path', '')}</p>
        <p><strong>模式:</strong> {s['mode']} | <strong>检测类别:</strong> {categories_str}</p>
        <p><strong>正样本:</strong> {s['positive_count']} | <strong>负样本:</strong> {s['negative_count']}</p>
        <p><strong>提示词:</strong></p>
        <div class="prompt">{s.get('prompt', 'N/A')}</div>
        <h4>📷 示例图片 (每类一张)</h4>
        <div class="sample-row">
"""
        samples = sample_images.get(exp_id, {})
        for cat in ["TP", "FP", "TN", "FN"]:
            if cat in samples:
                rel_path = samples[cat]
                img_full_path = os.path.join(exp_root, rel_path)
                img_base64 = image_to_base64(img_full_path)
                if img_base64:
                    img_src = f"data:image/jpeg;base64,{img_base64}"
                    html_content += f"""
            <div class="sample-card">
                <strong>{cat}</strong><br>
                <img src="{img_src}" alt="{cat} sample">
            </div>"""
                else:
                    html_content += f"""
            <div class="sample-card">
                <strong>{cat}</strong><br>
                <span style="color:gray;">图片读取失败</span>
            </div>"""
            else:
                html_content += f"""
            <div class="sample-card">
                <strong>{cat}</strong><br>
                <span style="color:gray;">无图片</span>
            </div>"""
        html_content += """
        </div>
    </div>
"""
    html_content += """
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML报告已生成: {html_path}")
    return html_path


def main():
    parser = argparse.ArgumentParser(description="批量目标检测实验脚本")
    parser.add_argument("--exp_root", type=str, default=None,
                        help="实验根目录。若不指定，则自动创建带时间戳的新目录。")
    parser.add_argument("--force", action="store_true",
                        help="强制覆盖已存在的实验子目录")
    args = parser.parse_args()

    # 确定实验根目录
    if args.exp_root:
        exp_root = args.exp_root
        os.makedirs(exp_root, exist_ok=True)
        print(f"使用指定实验目录: {exp_root}")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_root = os.path.join(DEFAULT_OUTPUT_ROOT, f"exp_{timestamp}")
        os.makedirs(exp_root, exist_ok=True)
        print(f"创建新实验目录: {exp_root}")

    # 加载已有记录
    existing_ids, existing_records = load_existing_summaries(exp_root)
    new_summaries = []

    for model_info in MODELS:
        model_name = model_info["name"]
        api_base = model_info.get("api_base", DEFAULT_API_BASE)

        for task_name, task_config in TASKS.items():
            dataset_path = task_config["path"]
            task_categories = task_config["categories"]

            if not os.path.isdir(dataset_path):
                print(f"警告: 数据集路径不存在，跳过: {dataset_path}")
                continue

            for prompt_key, prompt_text in PROMPTS.items():
                mode = "detection" if prompt_key.startswith("detection") else "vqa"
                exp_id = f"{model_name}_{task_name}_{prompt_key}"

                # 检查是否已存在
                if exp_id in existing_ids and not args.force:
                    print(f"实验 {exp_id} 已存在，跳过（使用 --force 可强制覆盖）")
                    continue

                out_dir = os.path.join(exp_root, exp_id)
                if os.path.exists(out_dir) and args.force:
                    import shutil
                    shutil.rmtree(out_dir)
                    print(f"强制删除已存在目录: {out_dir}")
                os.makedirs(out_dir, exist_ok=True)

                print(f"\n{'#'*60}")
                print(f"开始实验: {exp_id}")
                print(f"输出目录: {out_dir}")
                print(f"检测类别: {task_categories}")

                try:
                    stats, records = run_evaluation(
                        input_dir=dataset_path,
                        output_dir=out_dir,
                        categories=task_categories,
                        api_base=api_base,
                        model_name=model_name,
                        prompt_template=prompt_text,
                        mode=mode,
                        delay=1.0
                    )
                    stats["experiment_id"] = exp_id
                    stats["prompt_key"] = prompt_key
                    stats["task_name"] = task_name
                    stats["dataset_path"] = dataset_path
                    stats["categories"] = task_categories
                    new_summaries.append(stats)
                    existing_ids.add(exp_id)  # 更新已存在集合
                except Exception as e:
                    print(f"实验失败: {e}")
                    continue

    # 合并所有记录
    all_summaries = []
    for rec in existing_records:
        converted = {}
        for k, v in rec.items():
            if k in ["total_images", "positive_count", "negative_count", "TP", "TN", "FP", "FN"]:
                converted[k] = int(v) if v else 0
            elif k in ["recall", "precision", "f1", "accuracy", "miss_rate", "false_alarm", "total_time", "avg_time"]:
                converted[k] = float(v) if v else 0.0
            elif k == "categories":
                converted[k] = v.split("、") if v else []
            else:
                converted[k] = v
        all_summaries.append(converted)

    all_summaries.extend(new_summaries)

    if all_summaries:
        # 重新生成总汇总CSV
        summary_file = os.path.join(exp_root, "all_experiments_summary.csv")
        with open(summary_file, "w", newline="", encoding="utf-8-sig") as f:
            fieldnames = ["experiment_id", "model", "dataset", "task_name", "mode", "prompt_key",
                          "categories",
                          "total_images", "positive_count", "negative_count",
                          "TP", "TN", "FP", "FN",
                          "recall", "precision", "f1", "accuracy", "miss_rate", "false_alarm",
                          "total_time", "avg_time"]
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for s in all_summaries:
                s_csv = s.copy()
                if isinstance(s_csv.get("categories"), list):
                    s_csv["categories"] = "、".join(s_csv["categories"])
                writer.writerow(s_csv)

        # 重新生成HTML报告
        generate_html_report(exp_root, all_summaries)
        print(f"\n所有实验完成！结果保存在: {exp_root}")

if __name__ == "__main__":
    main()