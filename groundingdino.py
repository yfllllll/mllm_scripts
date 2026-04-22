import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from mmdet.apis import DetInferencer
import os

# --- 全局配置 ---
ADDITIONAL_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
    '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E2'
]

def plot_bounding_boxes(im, bounding_boxes, score_thr=0.05, bbox_size_thr=50.0):
    """绘制边界框"""
    img = im.copy()
    draw = ImageDraw.Draw(img)
    
    try:
        # 建议将字体路径设为环境变量或配置项
        font = ImageFont.truetype("./wqy-microhei.ttc", size=20)
    except:
        font = ImageFont.load_default()

    for i, bbox_info in enumerate(bounding_boxes):
        score = bbox_info.get("score", 0.0)
        if score < score_thr:
            continue
        
        color = ADDITIONAL_COLORS[i % len(ADDITIONAL_COLORS)]
        if "bbox_2d" not in bbox_info:
            continue
            
        x1, y1, x2, y2 = bbox_info["bbox_2d"]
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        # if bbox_width < bbox_size_thr or bbox_height < bbox_size_thr:
        #     continue
        if bbox_width * bbox_height < bbox_size_thr ** 2:
            continue
        draw.rectangle(((x1, y1), (x2, y2)), outline=color, width=4)
        
        label_text = bbox_info.get("label", "")
        # 绘制背景框让文字更清晰
        draw.rectangle(((x1, y1 - 25), (x1 + 150, y1)), fill=color)
        draw.text((x1 + 5, y1 - 22), label_text, fill="white", font=font)

    return img

class ChineseGroundingDetectionTab:
    def __init__(self) -> None:
        self.model_list = ['grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det', 'grounding_dino_base']
        self.model_info = {
            'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det': {
                'model': '/mnt/data/lyf/groundingdino/mmdetection-main/configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_all.py',
                'weights': '/mnt/data/lyf/groundingdino/grounding_dino_swin-l_pretrain_all-56d69e78.pth'
            },
            'grounding_dino_base': {
                'model': '/mnt/data/lyf/agent/gd/desk_test_pipeline1.py',  #/mnt/data/lyf/groundingdino/mmdetection-main/configs/mm_grounding_dino/grounding_dino_swin-l_pretrain_all.py',
                'weights': '/mnt/data/lyf/agent/gd/base_20251211_7.pth' #'/mnt/data/lyf/groundingdino/grounding_dino_swin-l_pretrain_all-56d69e78.pth'
            },
        }
        # --- 核心优化：存储已加载的模型实例 ---
        self.loaded_models = {} 
        self.create_ui()

    def get_model(self, model_name, device_index):
        """
        单例模式获取模型：如果模型已加载且设备一致，则直接返回
        """
        device = f"cuda:{int(device_index)}" if torch.cuda.is_available() else "cpu"
        model_key = f"{model_name}_{device}"
        
        if model_key not in self.loaded_models:
            print(f"--- 正在加载模型 {model_name} 到 {device} ---")
            # 清理旧显存
            if len(self.loaded_models) > 0:
                torch.cuda.empty_cache()
                
            self.loaded_models[model_key] = DetInferencer(
                **self.model_info[model_name],
                scope='mmdet',
                device=device
            )
        return self.loaded_models[model_key]

    def create_ui(self):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 中文目标检测演示")
                
                select_model = gr.Dropdown(
                    label='选择模型',
                    choices=self.model_list,
                    value=self.model_list[0],
                )
                
                # --- 新增：指定显卡输入 ---
                device_id = gr.Number(
                    label='GPU 设备 ID (如: 0, 1)',
                    value=6,
                    precision=0
                )
                
                image_input = gr.Image(label='上传图片', type='filepath')
                text_input = gr.Textbox(
                    label='输入中文类别名（用句号分隔）',
                    placeholder='例如：汽车. 行人. 自行车',
                    value='人. 车辆. 牛. 马. 熊. 羊. 驴. 鹿. 狼. 猪. 豹. 猴'
                )
                
                pred_score_thr = gr.Slider(
                    label='置信度阈值',
                    minimum=0., maximum=1.0, value=0.05, step=0.05
                )
                # 添加一个bbox size阈值过滤选项（可选）
                bbox_size_thr = gr.Slider(
                    label='bbox size阈值',
                    minimum=0., maximum=1000.0, value=50.0, step=10.0
                )
                
                run_button = gr.Button('开始检测', variant='primary')
                  
            with gr.Column(scale=1):
                output = gr.Image(label='检测结果', interactive=False)
                result_info = gr.Textbox(label='检测信息', interactive=False, lines=5)

        run_button.click(
            self.inference,
            inputs=[select_model, image_input, text_input, pred_score_thr, device_id, bbox_size_thr],
            outputs=[output, result_info]
        )

    def inference(self, model_name, image, text, score_thr, device_id, bbox_size_thr):
        try:
            # 1. 获取（或复用）模型实例
            inferencer = self.get_model(model_name, device_id)
            
            # 2. 推理
            results_dict = inferencer(
                image,
                texts=text,
                custom_entities=True,
                pred_score_thr=score_thr,
                return_vis=False
            )
            
            # 3. 解析结果
            predictions = results_dict['predictions'][0]
            bboxes = predictions.get('bboxes', [])
            labels = predictions.get('labels', [])
            scores = predictions.get('scores', [])
            
            original_image = Image.open(image).convert('RGB')
            class_names = [c.strip() for c in text.split('.')]
            
            bounding_boxes = []
            for bbox, label_idx, score in zip(bboxes, labels, scores):
                # 容错处理：确保标签索引不越界
                label_name = class_names[label_idx] if label_idx < len(class_names) else "Unknown"
                
                bounding_boxes.append({
                    "bbox_2d": bbox,
                    "label": f"{label_name}",
                    "score": float(score)
                })
            
            # 4. 可视化
            if bounding_boxes:
                vis_image = plot_bounding_boxes(original_image, bounding_boxes, score_thr, bbox_size_thr)
                result_text = f"成功检测到 {len(bounding_boxes)} 个目标"
            else:
                vis_image = original_image
                result_text = "未检测到任何目标"
              
            return vis_image, result_text
              
        except Exception as e:
            return None, f"推理出错: {str(e)}"

def create_demo():
    with gr.Blocks(title="MM-Grounding-DINO", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎯 MM-Grounding-DINO 中文目标检测")
        with gr.Tabs():
            with gr.TabItem("检测主界面"):
                ChineseGroundingDetectionTab()
    return demo

if __name__ == '__main__':
    demo = create_demo()
    # server_name="0.0.0.0" 允许外部访问
    demo.launch(server_name="0.0.0.0", server_port=9008)