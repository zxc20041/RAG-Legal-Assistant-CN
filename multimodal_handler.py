import base64
import requests
import os
import json
import hashlib
import hmac
import time
from datetime import datetime

# ================= 配置区 =================
# 1. 云雾 API 配置 (用于图片识别，已验证稳定)
YUNWU_API_KEY = "YOUR-YUNWU-API-KEY"
YUNWU_BASE_URL = "https://yunwu.ai/v1"
IMAGE_MODEL = "gpt-4o-mini"

# 2. 腾讯云配置 (用于音频识别，解决 429/503 问题)
# 申请地址：https://console.cloud.tencent.com/cam/capi
TENCENT_SECRET_ID = "YOUR-TENCENT-SECRET-ID"
TENCENT_SECRET_KEY = "YOUR-TENCENT-SECRET-KEY"
# ==========================================

def process_multimodal_file(file_path):
    if not os.path.exists(file_path):
        return f"错误：文件 {file_path} 不存在"

    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png']:
        return ocr_image_yunwu(file_path)
    elif ext in ['.mp3', '.wav', '.m4a']:
        return asr_audio_tencent(file_path)
    return "不支持的格式"

def ocr_image_yunwu(file_path):
    """图片解析逻辑"""
    try:
        with open(file_path, "rb") as f:
            base64_data = base64.b64encode(f.read()).decode('utf-8')
        headers = {"Authorization": f"Bearer {YUNWU_API_KEY}"}
        payload = {
            "model": IMAGE_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "提取文字，严禁代码。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_data}"}}
                ]
            }],
            "temperature": 0.0
        }
        res = requests.post(f"{YUNWU_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=60)
        return res.json()['choices'][0]['message']['content']
    except:
        return "图片解析异常"

def asr_audio_tencent(file_path):
    """
    腾讯云一句话识别 API (自动格式识别增强版)
    支持：wav, mp3, m4a, pcm 等
    """
    try:
        if not os.path.exists(file_path):
            return f"错误：文件 {file_path} 不存在"

        # 1. 读取音频并自动检测真实格式 (关键修改)
        with open(file_path, "rb") as f:
            audio_bytes = f.read()
            file_size = len(audio_bytes)
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # 检测前12个字节判断真实格式
            f.seek(0)
            header = f.read(12)
            
        if file_size == 0:
            return "错误：音频文件为空"

        # 逻辑：优先通过文件头判断，兜底才看后缀
        if b"ftyp" in header or b"isom" in header:
            real_format = "m4a"
        elif b"ID3" in header or b"\xff\xfb" in header:
            real_format = "mp3"
        elif b"RIFF" in header:
            real_format = "wav"
        else:
            # 如果识别不出，则回退到提取后缀名
            real_format = os.path.splitext(file_path)[1][1:].lower()
            if not real_format: real_format = "mp3" # 默认兜底

        # 2. 构造 API 基本信息
        service, host, region = "asr", "asr.tencentcloudapi.com", "ap-shanghai"
        action, version = "SentenceRecognition", "2019-06-14" 
        timestamp = int(time.time())
        date = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d")
        
        # 3. 构造请求体 (确保参数与实际文件头匹配)
        params = {
            "ProjectId": 0,
            "SubServiceType": 2,
            "EngSerViceType": "16k_zh", # 常见手机录音建议保持 16k_zh
            "SourceType": 1,
            "VoiceFormat": real_format, 
            "Data": audio_b64,
            "DataLen": file_size
        }
        payload = json.dumps(params, separators=(',', ':'))
        
        # 4. V3 签名核心逻辑
        def sign(key, msg): return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()
        
        canonical_headers = f"content-type:application/json\nhost:{host}\nx-tc-action:{action.lower()}\n"
        signed_headers = "content-type;host;x-tc-action"
        hashed_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        canonical_request = f"POST\n/\n\n{canonical_headers}\n{signed_headers}\n{hashed_payload}"
        
        credential_scope = f"{date}/{service}/tc3_request"
        string_to_sign = f"TC3-HMAC-SHA256\n{timestamp}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}"
        
        secret_date = sign(("TC3" + TENCENT_SECRET_KEY).encode("utf-8"), date)
        secret_service = sign(secret_date, service)
        secret_key = sign(secret_service, "tc3_request")
        signature = hmac.new(secret_key, string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()
        
        # 5. 发送请求
        headers = {
            "Authorization": f"TC3-HMAC-SHA256 Credential={TENCENT_SECRET_ID}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}",
            "Content-Type": "application/json",
            "Host": host,
            "X-TC-Action": action,
            "X-TC-Timestamp": str(timestamp),
            "X-TC-Version": version,
            "X-TC-Region": region
        }
        
        resp = requests.post(f"https://{host}", headers=headers, data=payload.encode('utf-8'), timeout=60)
        res_json = resp.json()
        
        if "Response" in res_json:
            if "Error" in res_json["Response"]:
                # 针对常见错误的友好提示
                err_msg = res_json["Response"]["Error"]["Message"]
                return f"腾讯云识别报错: {err_msg} (检测格式: {real_format})"
            return res_json["Response"].get("Result", "音频未识别到文字")
        return "响应解析失败"
            
    except Exception as e:
        return f"系统级异常: {str(e)}"