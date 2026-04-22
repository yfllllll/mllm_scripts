import gradio as gr
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import torch
from ultralytics.models.sam import SAM3SemanticPredictor

import cv2
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class SAM3Detector:
    """SAM3开集目标检测器"""
    
    def __init__(self, model_path="sam3.pt"):
        """
        初始化SAM3检测器
        
        Args:
            model_path: SAM3模型路径
        """
        print("正在加载SAM3模型...")
        overrides = dict(
                conf=0.25,
                task="segment",
                mode="predict",
                model=model_path,
                imgsz=1288,
                half=True,  # Use FP16 for faster inference
                save=False,
            )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)
        print("SAM3模型加载完成！")

    def detect(self, image, categories, conf_threshold):
        """
        检测图像中的对象
        
        Args:
            image: filepath
            categories: 类别列表，如['cat', 'dog']
            conf_threshold: 置信度阈值
            
        Returns:
            result_image: 带检测框的图像
        """
        
        # 将categories转换为文本提示格式
        # text_prompt = ' . '.join(categories) + ' .'
        
        try:
            # 执行SAM3推理
            # self.predictor.set_image(image)
            
            results = self.predictor(
                source=image,
                text=categories,
                conf=conf_threshold,
            )
            
            # 创建输出图像,变换三通道顺序
            result_image = cv2.imread(image)
            result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            for result in results:  
                if result.boxes is not None:  
                    # 获取边界框信息 [1](#7-0)   
                    xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4) - x1, y1, x2, y2  
                    conf = result.boxes.conf.cpu().numpy()  # (N, 1) - 置信度  
                    cls = result.boxes.cls.cpu().numpy()    # (N, 1) - 类别ID  
                    
                    # 从 result.names 获取类别名称映射 [2](#7-1)   
                    names = result.names  
                    for i in range(len(xyxy)):
                        if conf[i] >= conf_threshold:  # 使用固定阈值0.35
                            x1, y1, x2, y2 = map(int, xyxy[i])
                            label = names[int(cls[i])]
                            area = (x2 - x1) * (y2 - y1)
                            
                            # 绘制边界框和标签
                            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(result_image, f"{label} {conf[i]:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                           
                return result_image

        except Exception as e:
            print(f"检测失败: {e}")
            return result_image

    def detect_from_image(self, image, categories, conf):

        return self.detect(image, categories, conf)


def create_gradio_demo():
    """创建Gradio演示界面"""
    
    # 初始化检测器
    detector = SAM3Detector(model_path="/mnt/data/lyf/sam3/sam3/sam3.pt")
    
    # 示例类别
    example_categories = [
        ["person", "car", "tree"],
        ["dog", "cat"],
        ["chair", "table", "sofa"],
        ["apple", "banana", "orange"]
    ]
    
    # 示例图片URLs
    example_images = [
        "https://images.unsplash.com/photo-1506744038136-46273834b3fb",
        "https://images.unsplash.com/photo-1514888286974-6d03bde4ba2f",
        "https://images.unsplash.com/photo-1517336714731-489689fd1ca8",
        "https://images.unsplash.com/photo-1544568100-847a948585b9"
    ]
    
    def process_image(image, categories_text, conf):
        """处理图像的主函数"""
        # 解析类别文本
        categories = [cat.strip() for cat in categories_text.split(",") if cat.strip()]
        if not categories:
            return image, "请至少输入一个类别"
        
        # 执行检测
        result_img = detector.detect(image, categories, conf)

        # 创建结果文本
        # if boxes_info:
        #     result_text = f"检测到 {len(boxes_info)} 个对象：\n"
        #     for i, box in enumerate(boxes_info, 1):
        #         result_text += f"{i}. {box['label']}: 位置({box['bbox'][0]:.0f},{box['bbox'][1]:.0f})-({box['bbox'][2]:.0f},{box['bbox'][3]:.0f}), 面积:{box['area']:.0f}像素\n"
        # else:
        #     result_text = "未检测到对象"

        return result_img

    # 创建Gradio界面
    with gr.Blocks(title="SAM3 开集目标检测演示", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🚀 SAM3 开集目标检测演示
        
        SAM3（Segment Anything Model 3）是一个强大的开集目标检测和分割模型。
        输入图像和类别名称，模型会自动检测这些类别的对象。
        
        **使用说明：**
        1. 上传或拖拽图像
        2. 输入要检测的类别，用逗号分隔（例如：person, car, tree）
        3. 调整置信度阈值（默认0.35）
        4. 点击"开始检测"按钮
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # 输入组件
                image_input = gr.Image(
                    label="输入图像",
                    type="filepath",
                    height=600
                )
                
                categories_input = gr.Textbox(
                    label="检测类别（用逗号分隔）",
                    placeholder="例如: person, car, dog, tree",
                    value="person, car"
                )
                
                conf_slider = gr.Slider(
                    label="置信度阈值",
                    minimum=0.05,
                    maximum=0.9,
                    value=0.55,
                    step=0.05
                )
                
                detect_btn = gr.Button(
                    "开始检测",
                    variant="primary",
                    size="lg"
                )
                
                # 示例
                with gr.Accordion("💡 使用示例", open=False):
                    gr.Examples(
                        examples=[
                            ["person, car", 0.35],
                            ["dog, cat", 0.3],
                            ["chair, table, sofa", 0.4],
                            ["apple, banana, orange", 0.35]
                        ],
                        inputs=[categories_input, conf_slider],
                        label="快速设置"
                    )
            
            with gr.Column(scale=1):
                # 输出组件
                image_output = gr.Image(
                    label="检测结果",
                    height=600
                )
                
                # result_output = gr.Textbox(
                #     label="检测结果详情",
                #     lines=8
                # )
        
        # 按钮点击事件
        detect_btn.click(
            fn=process_image,
            inputs=[image_input, categories_input, conf_slider],
            outputs=[image_output]
        )
        
        # 底部信息
        gr.Markdown("""
        ---
        **技术说明：**
        - 使用SAM3模型进行开集目标检测
        - 支持任意文本类别输入
        - 基于概念分割生成边界框
        - 检测面积阈值: 2500像素
        """)
    
    return demo, detector


if __name__ == "__main__":
    # 创建演示界面
    demo, detector = create_gradio_demo()
    
    # 启动Gradio应用
    print("🎉 SAM3 Gradio演示已启动！")
    print("📱 请在浏览器中访问: http://localhost:9008")
    print("📋 使用说明:")
    print("   1. 上传图像")
    print("   2. 输入要检测的类别，用逗号分隔")
    print("   3. 调整置信度阈值")
    print("   4. 点击'开始检测'按钮")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=9008,
        share=False,  # 设置为True可以创建公共链接
        debug=True
    )