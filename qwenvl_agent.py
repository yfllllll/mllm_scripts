import io
import json5
import base64
import asyncio
import re
import imghdr
import logging
from typing import List, Optional
from functools import partial

import httpx
from PIL import Image, ImageDraw
from fastapi import FastAPI, Form, HTTPException, Body, File, UploadFile
from pydantic import BaseModel

# qwen_agent 相关导入（请根据实际安装版本调整）
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.llm.schema import Message, ContentItem

# --------------------------
# 配置日志（便于调试）
# --------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------------------------
# 1. 远程 Grounding DINO 工具（增强健壮性）
# --------------------------
@register_tool('grounding_dino_detect')
class GroundingDinoRemoteTool(BaseTool):
    description = "调用远程 Grounding DINO 服务检测图像中的目标，返回边界框坐标、类别与置信度"
    parameters = {
        "type": "object",
        "properties": {
            "image_base64": {"type": "string", "description": "base64 编码的图像数据（不带 data:image 前缀）"},
            "categories": {"type": "array", "items": {"type": "string"}, "description": "待检测类别列表", "default": []}
        },
        "required": ["image_base64"]
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)
        # 从配置中读取服务地址，支持环境变量覆盖
        self.service_url = cfg.get("service_url", "http://localhost:1006/detect_objects") if cfg else "http://localhost:1006/detect_objects"

    @staticmethod
    def _clean_base64(b64: str) -> str:
        """清理 base64 字符串：去除 data:image 前缀及所有空白字符"""
        if ',' in b64:
            b64 = b64.split(',', 1)[1]
        # 移除空白字符（空格、换行、制表符等）
        b64 = re.sub(r'\s', '', b64)
        return b64

    def call(self, params: str, **kwargs) -> str:
        """执行工具调用，返回 JSON 格式结果"""
        try:
            args = json5.loads(params)
            raw_b64 = args.get("image_base64")
            if not raw_b64:
                return json5.dumps({"status": "error", "msg": "缺少 image_base64 参数"}, ensure_ascii=False)

            # 核心修复：强制清洗 base64
            image_b64 = self._clean_base64(raw_b64)
            categories = args.get("categories", [])

            # 细化超时设置
            timeout = httpx.Timeout(10.0, read=30.0)
            with httpx.Client(timeout=timeout) as client:
                resp = client.post(
                    self.service_url,
                    json={"image_base64": image_b64, "categories": categories}
                )
                resp.raise_for_status()
                data = resp.json()

            if not data.get("success"):
                error_msg = data.get("error", "远程检测失败（未知错误）")
                return json5.dumps({"status": "error", "msg": error_msg}, ensure_ascii=False)

            # 验证并提取边界框
            boxes = []
            for b in data.get("boxes", []):
                # 确保必要字段存在
                if all(k in b for k in ("xmin", "ymin", "xmax", "ymax", "label")):
                    boxes.append({
                        "xmin": float(b["xmin"]),
                        "ymin": float(b["ymin"]),
                        "xmax": float(b["xmax"]),
                        "ymax": float(b["ymax"]),
                        "label": b["label"],
                        "confidence": float(b.get("confidence", 1.0))
                    })
                else:
                    logger.warning(f"远程服务返回的框缺少必要字段: {b}")

            return json5.dumps({"status": "success", "boxes": boxes}, ensure_ascii=False)

        except httpx.TimeoutException:
            logger.error("调用远程 DINO 服务超时")
            return json5.dumps({"status": "error", "msg": "远程检测服务超时"}, ensure_ascii=False)
        except httpx.HTTPStatusError as e:
            logger.error(f"远程 DINO 服务返回错误状态码: {e.response.status_code}")
            return json5.dumps({"status": "error", "msg": f"远程服务返回错误: {e.response.status_code}"}, ensure_ascii=False)
        except Exception as e:
            logger.exception("调用远程 DINO 服务发生未知异常")
            return json5.dumps({"status": "error", "msg": f"调用远程服务失败: {str(e)}"}, ensure_ascii=False)

# --------------------------
# 2. 初始化 Assistant
# --------------------------
def load_assistant():
    llm_cfg = {
        'model': 'Qwen3-VL-4B-Instruct',
        'model_type': 'qwenvl_oai',
        'model_server': 'http://localhost:8000/v1',  # 请根据实际服务地址修改
        'api_key': 'EMPTY',
        'generate_cfg': {'top_p': 0.8, 'temperature': 0.7}
    }
    assistant = Assistant(
        llm=llm_cfg,
        system_message="你是具备视觉理解能力的多模态助手。当用户需要检测物体时，调用 grounding_dino_detect 工具，传入图像的 base64 编码和待检测类别；其他问题直接回答。",
        function_list=['grounding_dino_detect']
    )
    return assistant

assistant = load_assistant()

# --------------------------
# 3. FastAPI 响应模型
# --------------------------
class DetectResponse(BaseModel):
    status: str
    msg: str
    detect_img_base64: Optional[str] = None
    boxes: Optional[List[dict]] = None
    answer_text: Optional[str] = None

class DetectRequest(BaseModel):
    image_base64: str
    query: str

# --------------------------
# 4. 辅助函数
# --------------------------
def _draw_boxes(image_b64: str, boxes: List[dict]) -> str:
    """在图像上绘制边界框，返回标注后的 base64 JPEG 字符串"""
    img_data = base64.b64decode(image_b64)
    pil_img = Image.open(io.BytesIO(img_data)).convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    for b in boxes:
        # 转换为整数坐标，避免 PIL 警告
        xmin = int(b["xmin"])
        ymin = int(b["ymin"])
        xmax = int(b["xmax"])
        ymax = int(b["ymax"])
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin, ymin - 15), f"{b['label']} {b['confidence']:.2f}", fill="red")
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def _clean_base64_for_llm(b64: str) -> str:
    """清洗 base64 用于构造 ContentItem（只去除空白，保留前缀由调用方决定）"""
    return re.sub(r'\s', '', b64)

# --------------------------
# 5. 核心 Agent 调用（同步版本，将被放入线程池）
# --------------------------
def _run_assistant_sync(image_base64: str, query: str, mime: str = "image/jpeg"):  
    clean_b64 = _clean_base64_for_llm(image_base64)  
    image_uri = f"data:{mime};base64,{clean_b64}"  
    content = [ContentItem(text=query), ContentItem(image=image_uri)]  
    messages = [Message(role="user", content=content)]  
  
    # 使用 run_nonstream 获取完整回复  
    responses = assistant.run_nonstream(messages=messages)  # 返回 List[Message]  
  
    tool_called = False  
    boxes = None  
    tool_error = None  
    final_text = ""  
  
    for msg in responses:  
        role = getattr(msg, "role", "")  
        content = getattr(msg, "content", "")  
        func_call = getattr(msg, "function_call", None)  
  
        if role == "assistant":  
            if func_call:  
                tool_called = True  
            if isinstance(content, str) and content:  
                final_text = content  # 取最后一条非空文本  
        elif role == "function" and content:  
            try:  
                res = json5.loads(content)  
                if res.get("status") == "success":  
                    boxes = res.get("boxes", [])  
                else:  
                    tool_error = res.get("msg", "工具调用失败")  
            except Exception as e:  
                logger.exception("解析工具返回数据失败")  
                tool_error = f"工具返回格式错误: {str(e)}"  
  
    return tool_called, boxes, tool_error, final_text

async def _run_assistant(image_base64: str, query: str, mime: str = "image/jpeg"):
    """异步包装，将同步函数放入线程池，避免阻塞事件循环"""
    loop = asyncio.get_event_loop()
    func = partial(_run_assistant_sync, image_base64, query, mime)
    return await loop.run_in_executor(None, func)

# --------------------------
# 6. FastAPI 应用
# --------------------------
app = FastAPI(title="Qwen3-VL + Grounding DINO 智能体服务（修复版）")

@app.post("/detect", response_model=DetectResponse)
async def detect(
    image_base64: Optional[str] = Form(None),
    query: Optional[str] = Form(None),
    json_req: Optional[DetectRequest] = Body(None)
):
    """
    接收 JSON 或表单格式的请求，支持带前缀或不带前缀的 base64。
    注意：无法自动检测 MIME 类型，默认使用 image/jpeg，若图像格式非 JPEG 可能导致 LLM 解析失败。
    建议使用 /detect_file 上传文件。
    """
    if json_req is not None:
        image_base64 = json_req.image_base64
        query = json_req.query
    elif image_base64 is None or query is None:
        raise HTTPException(
            status_code=400,
            detail="请求缺少必要参数：请以 JSON 格式提供 {'image_base64': '...', 'query': '...'} 或以表单形式提供 image_base64 和 query 字段"
        )

    try:
        # 默认 MIME 为 image/jpeg（无法从字符串检测真实格式）
        mime = "image/jpeg"
        tool_called, boxes, tool_error, final_text = await _run_assistant(image_base64, query, mime)

        if tool_called:
            if tool_error:
                logger.error(f"工具调用失败: {tool_error}")
                raise HTTPException(status_code=500, detail=f"目标检测服务异常: {tool_error}")
            # 无论是否有框，均返回检测结果
            annotated_b64 = _draw_boxes(image_base64, boxes) if boxes else None
            return DetectResponse(
                status="success",
                msg=f"检测到 {len(boxes)} 个目标" if boxes else "未检测到任何目标",
                detect_img_base64=annotated_b64,
                boxes=boxes or []
            )
        else:
            return DetectResponse(
                status="success",
                msg="问答完成",
                answer_text=final_text
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("/detect 处理异常")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.post("/detect_file", response_model=DetectResponse)
async def detect_file(
    file: UploadFile = File(...),
    query: str = Form(...)
):
    """
    上传图像文件进行检测或问答。
    自动识别图像格式，构造正确的 MIME 类型，避免编码错误。
    """
    try:
        image_bytes = await file.read()
        # 检测真实图像格式
        img_type = imghdr.what(None, h=image_bytes)
        if not img_type:
            # 回退：尝试用 PIL 识别
            try:
                with Image.open(io.BytesIO(image_bytes)) as img:
                    img_type = img.format.lower()
            except Exception:
                img_type = 'jpeg'  # 最终默认
        if img_type == 'jpeg':
            mime = 'image/jpeg'
        elif img_type == 'png':
            mime = 'image/png'
        elif img_type == 'webp':
            mime = 'image/webp'
        else:
            mime = f'image/{img_type}'  # 其他格式尽量保留

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        tool_called, boxes, tool_error, final_text = await _run_assistant(image_base64, query, mime)

        if tool_called:
            if tool_error:
                logger.error(f"工具调用失败: {tool_error}")
                raise HTTPException(status_code=500, detail=f"目标检测服务异常: {tool_error}")
            annotated_b64 = _draw_boxes(image_base64, boxes) if boxes else None
            return DetectResponse(
                status="success",
                msg=f"检测到 {len(boxes)} 个目标" if boxes else "未检测到任何目标",
                detect_img_base64=annotated_b64,
                boxes=boxes or []
            )
        else:
            return DetectResponse(
                status="success",
                msg="问答完成",
                answer_text=final_text
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("/detect_file 处理异常")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

# --------------------------
# 7. 启动入口
# --------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9020)