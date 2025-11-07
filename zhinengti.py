import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
import time
from datetime import datetime
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import json

# --- matplotlib 中文字体配置 ---
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 1. 重新定义工具和模型 ---

class ReportInput(BaseModel):
    """用于生成详细故障诊断报告的工具"""
    fault_id: str = Field(description="当前故障事件的唯一标识ID,例如:'EVENT-20251028-001'")
    severity: str = Field(description="故障的严重程度,例如:'一级预警'或'二级预警'")
    fault_type: str = Field(description="故障类型,如:'串联电弧故障'、'绝缘老化'等")

class StabilityInput(BaseModel):
    """用于查询船端边缘计算单元和船岸协同通信链路的实时状态和负载率"""
    pass

class MaintenanceInput(BaseModel):
    """根据故障类型生成维护工单"""
    circuit_id: str = Field(description="回路编号,例如:'03号舱回路'")
    fault_severity: str = Field(description="故障严重程度")
    maintenance_type: str = Field(description="维护类型:预防性/紧急")

def generate_diagnostic_report(fault_id: str, severity: str, fault_type: str) -> str:
    """生成格式化的故障诊断报告"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_data = {
        "report_id": f"RPT-{fault_id}",
        "timestamp": timestamp,
        "fault_severity": severity,
        "fault_type": fault_type,
        "dl_confidence": "97.5%",
        "root_cause": "高振动区域电缆固定件老化松动导致的串联电弧故障",
        "maintenance_advice": "立即进行预防性检查,紧固连接件,参考CCS规范第5.4.1条",
        "risk_level": "高" if "二级" in severity else "中"
    }
    return f"【诊断报告】{json.dumps(report_data, ensure_ascii=False, indent=2)}"

def check_system_stability() -> str:
    """查询系统稳定性状态"""
    stability_data = {
        "edge_compute_load": "38%",
        "inference_latency": "15ms",
        "communication_latency": "45ms",
        "model_accuracy": "97.5%",
        "system_status": "稳定"
    }
    return f"【系统状态】{json.dumps(stability_data, ensure_ascii=False)}"

def generate_maintenance_order(circuit_id: str, fault_severity: str, maintenance_type: str) -> str:
    """生成维护工单"""
    order_data = {
        "order_id": f"MO-{datetime.now().strftime('%Y%m%d%H%M')}",
        "circuit": circuit_id,
        "maintenance_type": maintenance_type,
        "priority": "紧急" if "二级" in fault_severity else "高",
        "required_tools": "红外热像仪,力矩扳手,绝缘测试仪",
        "estimated_duration": "2小时",
        "safety_requirements": "断电操作,穿戴PPE"
    }
    return f"【维护工单】{json.dumps(order_data, ensure_ascii=False)}"

# --- 2. 修复智能体响应函数 ---

@st.cache_resource
def get_gemini_client():
    """安全地获取 Gemini 客户端"""
    try:
        if "gemini_api_key" not in st.secrets:
            return None 
        GEMINI_API_KEY = st.secrets["gemini_api_key"]
        return genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"初始化 Gemini 客户端失败: {e}")
        return None

def gemini_agent_response(user_query: str, system_status: dict):
    """完全重写的智能体响应函数"""
    client = get_gemini_client()
    
    if client is None:
        return "⚠️ Gemini 客户端未初始化（可能缺少 API Key），无法执行 AI 推理。请检查配置。"
    
    # 构建详细的系统状态上下文
    status_context = (
        f"【实时系统状态 - {system_status['timestamp']}】\n"
        f"- 检测状态: {system_status['detection_status']}\n"
        f"- 模型置信度: {system_status['confidence']:.1f}%\n" 
        f"- 故障类型: {system_status['fault_type']}\n"
        f"- 监测回路: {system_status['circuit_id']}\n"
        f"- 当前场景: {st.session_state.fault_scenario}\n"
    )
    
    # 知识库增强
    GROUNDING_FACTS = (
        "【船舶电气安全知识库】\n"
        "**预警等级说明:**\n"
        "- 一级预警: 早期风险，需要预防性维护\n"
        "- 二级预警: 严重故障，需要紧急处理\n"
        "**可用工具:**\n"
        "- generate_diagnostic_report: 生成详细故障诊断报告\n" 
        "- check_system_stability: 检查系统健康状态\n"
        "- generate_maintenance_order: 生成维护工单\n"
        "**操作指南:** 对于专业问题请优先使用上述工具。"
    )

    # 明确的系统指令
    system_instruction = (
        "你是专业的船舶电气安全智能体。请遵循以下规则:\n"
        "1. 当用户询问故障分析、诊断报告时，使用 generate_diagnostic_report 工具\n"
        "2. 当用户询问系统状态、健康度时，使用 check_system_stability 工具\n" 
        "3. 当用户询问维护指导、工单时，使用 generate_maintenance_order 工具\n"
        "4. 优先使用工具处理专业问题，工具无法处理时再自主回答\n"
        "5. 回答要专业、准确、基于实时系统状态"
    )
    
    full_prompt = f"{system_instruction}\n\n{GROUNDING_FACTS}\n\n{status_context}\n\n用户提问: {user_query}"

    try:
        # 创建工具声明
        tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="generate_diagnostic_report",
                        description="生成详细的故障诊断报告，包含根本原因分析和维护建议",
                        parameters=ReportInput.model_json_schema()
                    ),
                    types.FunctionDeclaration(
                        name="check_system_stability",
                        description="查询船端边缘计算单元和船岸协同通信链路的实时状态",
                        parameters=StabilityInput.model_json_schema()
                    ),
                    types.FunctionDeclaration(
                        name="generate_maintenance_order", 
                        description="根据故障类型和严重程度生成具体的维护工单",
                        parameters=MaintenanceInput.model_json_schema()
                    )
                ]
            )
        ]

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=tools,
        )
        
        # 第一次调用 - 让模型决定是否使用工具
        response = client.models.generate_content(
            model='gemini-2.0-flash',  # 使用更稳定的版本
            contents=full_prompt,
            config=config,
        )
        
        # 检查是否有工具调用
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call'):
                            # 检测到工具调用
                            function_call = part.function_call
                            tool_name = function_call.name
                            
                            st.info(f"🔧 智能体正在使用工具: {tool_name}")
                            
                            # 根据工具名称准备参数
                            if tool_name == "generate_diagnostic_report":
                                tool_args = {
                                    'fault_id': f"EVENT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                                    'severity': system_status['detection_status'],
                                    'fault_type': system_status['fault_type']
                                }
                            elif tool_name == "check_system_stability":
                                tool_args = {}
                            elif tool_name == "generate_maintenance_order":
                                tool_args = {
                                    'circuit_id': system_status['circuit_id'],
                                    'fault_severity': system_status['detection_status'],
                                    'maintenance_type': "紧急" if "二级" in system_status['detection_status'] else "预防性"
                                }
                            else:
                                return f"未知工具: {tool_name}"
                            
                            # 执行工具
                            try:
                                tool_function = globals().get(tool_name)
                                if tool_function:
                                    tool_result = tool_function(**tool_args)
                                    
                                    # 第二次调用 - 让模型基于工具结果生成最终响应
                                    final_response = client.models.generate_content(
                                        model='gemini-2.0-flash',
                                        contents=[
                                            f"用户问题: {user_query}",
                                            f"系统状态: {status_context}",
                                            f"工具执行结果: {tool_result}",
                                            "请基于工具执行结果生成专业的回答:"
                                        ],
                                        config=types.GenerateContentConfig(
                                            system_instruction=system_instruction
                                        ),
                                    )
                                    return final_response.text
                                else:
                                    return f"工具函数 {tool_name} 不存在"
                                    
                            except Exception as tool_error:
                                return f"❌ 工具执行错误: {str(tool_error)}"
        
        # 如果没有工具调用，返回模型的直接响应
        return response.text

    except Exception as e:
        error_msg = f"智能体 API 调用失败: {str(e)}"
        st.error(error_msg)
        
        # 提供基于系统状态的简单回退响应
        if "诊断" in user_query or "报告" in user_query:
            return generate_diagnostic_report(
                f"FALLBACK-{datetime.now().strftime('%Y%m%d%H%M')}",
                system_status['detection_status'],
                system_status['fault_type']
            )
        elif "维护" in user_query or "工单" in user_query:
            return generate_maintenance_order(
                system_status['circuit_id'],
                system_status['detection_status'],
                "紧急" if "二级" in system_status['detection_status'] else "预防性"
            )
        elif "状态" in user_query or "健康" in user_query:
            return check_system_stability()
        else:
            return f"当前系统状态: {system_status['detection_status']} (置信度: {system_status['confidence']:.1f}%)"

# --- 3. 其他函数保持不变 ---

def simulate_current_data(t, fault_scenario="normal", prediction_mode=False):
    """模拟电流数据（保持不变）"""
    # ... 保持原有代码不变
    base_frequency = 50
    phase_offset = time.time() * 2 * np.pi * base_frequency / 1000 
    
    time_series = np.linspace(0, 2 / base_frequency, t)
    current = 10 * np.sin(2 * np.pi * base_frequency * time_series + phase_offset)
    current += np.random.normal(0, 0.05, t)
    
    if fault_scenario == "early_arc":
        mask = (time_series % 0.1 < 0.02)
        high_freq = np.sin(2 * np.pi * 5000 * time_series) * 0.3
        current += high_freq * mask
        
    elif fault_scenario == "severe_arc":
        high_freq = np.sin(2 * np.pi * 3000 * time_series) * 0.8
        current += high_freq + 2 * np.random.rand(t)
        
    elif fault_scenario == "motor_start":
        startup_effect = 3 * np.exp(-time_series * 2) * np.sin(2 * np.pi * 100 * time_series)
        current += startup_effect

    if prediction_mode:
        trend_factor = (time.time() - st.session_state.last_update) / 10 
        trend = 0.5 * np.exp(-time_series * 3) * np.sin(2 * np.pi * 150 * time_series) * (1 + trend_factor)
        current += trend

    return time_series * 1000, current

def dl_model_inference(data, fault_scenario):
    """模型推理（保持不变）"""
    if fault_scenario == "severe_arc":
        return "二级预警 (故障确认)", 97.5, "severe_arc"
    elif fault_scenario == "early_arc":
        if 'early_arc_confidence' not in st.session_state:
             st.session_state.early_arc_confidence = 70.0
        st.session_state.early_arc_confidence = min(90.0, st.session_state.early_arc_confidence + 0.5) 
        if st.session_state.early_arc_confidence > 70.0:
            return "一级预警 (预测风险)", st.session_state.early_arc_confidence, "early_arc"
        else:
            return "运行正常 (安全)", 5.0, "normal"
    elif fault_scenario == "motor_start":
        return "干扰信号 (电机启动)", 10.0, "motor_start"
    else:
        st.session_state.early_arc_confidence = 70.0 if 'early_arc_confidence' in st.session_state else 70.0
        return "运行正常 (安全)", 2.0, "normal"

# --- 4. 主界面（保持不变）---
def main():
    # ... 主界面代码保持不变
    # 确保使用修复后的 gemini_agent_response 函数

if __name__ == "__main__":
    main()
