import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import json

# --- 0. 环境和工具定义 ---

class ReportInput(BaseModel):
    """用于生成详细故障诊断报告的工具"""
    fault_id: str = Field(description="当前故障事件的唯一标识ID")
    severity: str = Field(description="故障的严重程度")
    fault_type: str = Field(description="故障类型")

class StabilityInput(BaseModel):
    """用于查询系统状态"""

class MaintenanceInput(BaseModel):
    """根据故障类型生成维护工单"""
    circuit_id: str = Field(description="回路编号")
    fault_severity: str = Field(description="故障严重程度")
    maintenance_type: str = Field(description="维护类型")

def generate_diagnostic_report(fault_id: str, severity: str, fault_type: str) -> str:
    """生成格式化的故障诊断报告"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_data = {
        "report_id": f"RPT-{fault_id}",
        "timestamp": timestamp,
        "fault_severity": severity,
        "fault_type": fault_type,
        "dl_confidence": "97.5%",
        "root_cause": "高振动区域电缆固定件老化松动",
        "maintenance_advice": "立即进行预防性检查，紧固连接件",
        "risk_level": "高" if "二级" in severity else "中"
    }
    return json.dumps(report_data, ensure_ascii=False, indent=2)

def check_system_stability() -> str:
    """查询系统稳定性状态"""
    stability_data = {
        "edge_compute_load": "38%",
        "inference_latency": "15ms", 
        "communication_latency": "45ms",
        "model_accuracy": "97.5%",
        "system_status": "稳定",
        "last_maintenance": "2025-01-15",
        "next_scheduled": "2025-04-15"
    }
    return json.dumps(stability_data, ensure_ascii=False)

def generate_maintenance_order(circuit_id: str, fault_severity: str, maintenance_type: str) -> str:
    """生成维护工单"""
    order_data = {
        "order_id": f"MO-{datetime.now().strftime('%Y%m%d%H%M')}",
        "circuit": circuit_id,
        "maintenance_type": maintenance_type,
        "priority": "紧急" if "二级" in fault_severity else "高",
        "required_tools": "红外热像仪,力矩扳手,绝缘测试仪",
        "estimated_duration": "2小时",
        "safety_requirements": "断电操作，穿戴PPE",
        "assigned_technician": "待分配"
    }
    return json.dumps(order_data, ensure_ascii=False)

AVAILABLE_TOOLS = {
    "generate_diagnostic_report": generate_diagnostic_report,
    "check_system_stability": check_system_stability,
    "generate_maintenance_order": generate_maintenance_order,
}

# --- 1. 数据模拟 ---
def simulate_current_data(t, fault_scenario="normal", prediction_mode=False):
    base_frequency = 50
    time_series = np.linspace(0, 2 / base_frequency, t)
    current = 10 * np.sin(2 * np.pi * base_frequency * time_series)
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
        trend = 0.5 * np.exp(-time_series * 3) * np.sin(2 * np.pi * 150 * time_series)
        current += trend

    return time_series * 1000, current

# --- 2. 模型推理模拟 ---
def dl_model_inference(data, fault_scenario):
    max_current = np.max(np.abs(data))
    high_freq_energy = np.std(data - np.mean(data))
    
    if fault_scenario == "severe_arc":
        return "二级预警 (故障确认)", 97.5, "severe_arc"
    elif fault_scenario == "early_arc":
        if high_freq_energy > 0.4:
            return "一级预警 (预测风险)", 85.0, "early_arc"
        else:
            return "运行正常 (安全)", 5.0, "normal"
    elif fault_scenario == "motor_start":
        return "干扰信号 (电机启动)", 10.0, "motor_start"
    else:
        return "运行正常 (安全)", 2.0, "normal"

# --- 3. 智能体核心逻辑 ---
@st.cache_resource
def get_gemini_client():
    try:
        GEMINI_API_KEY = st.secrets["gemini_api_key"]
        return genai.Client(api_key=GEMINI_API_KEY)
    except KeyError:
        st.error("初始化失败：无法找到 Gemini API 密钥。")
        st.stop()
    except Exception as e:
        st.error(f"初始化 Gemini 客户端失败: {e}")
        st.stop()

def gemini_agent_response(user_query: str, system_status: dict, conversation_history: list):
    """完全基于Gemini生成自然对话的智能体"""
    client = get_gemini_client()
    
    # 构建系统指令
    system_instruction = f"""
你是一个专业的船舶电气安全智能专家，名字是"海安"。请以专业、自然且友好的方式与用户对话。

当前系统状态：
- 监测回路：{system_status['circuit_id']}
- 检测状态：{system_status['detection_status']}
- 置信度：{system_status['confidence']}%
- 故障类型：{system_status['fault_type']}
- 更新时间：{system_status['timestamp']}

你的能力包括：
1. 实时监测与故障诊断
2. 趋势预测与预警
3. 生成诊断报告
4. 检查系统稳定性
5. 创建维护工单

对话风格要求：
- 专业但不过于技术化
- 友好且乐于助人
- 根据状态调整语气（正常时轻松，预警时关切，故障时紧急）
- 自然引用相关知识，不要生硬列出条款
- 使用工具时要解释其目的和结果意义

请基于当前系统状态和用户问题，提供专业、有用的回答。
"""
    
    try:
        # 构建对话历史
        history_text = ""
        for msg in conversation_history[-6:]:  # 只保留最近6轮对话
            role = "用户" if msg["role"] == "user" else "助手"
            history_text += f"{role}: {msg['content']}\n"
        
        # 构建完整的提示词
        full_prompt = f"{system_instruction}\n\n对话历史：\n{history_text}\n用户: {user_query}\n助手:"
        
        config = types.GenerateContentConfig(
            tools=list(AVAILABLE_TOOLS.values()),
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingMode.ANY
                )
            )
        )
        
        # 生成响应
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=full_prompt,
            config=config,
        )
        
        # 处理工具调用
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                # 检查是否有工具调用
                if hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call'):
                            function_call = part.function_call
                            tool_name = function_call.name
                            tool_args = dict(function_call.args)
                            
                            if tool_name in AVAILABLE_TOOLS:
                                # 执行工具调用
                                tool_result = AVAILABLE_TOOLS[tool_name](**tool_args)
                                
                                # 基于工具结果生成最终回复
                                tool_prompt = f"""
系统状态：{system_status}
用户问题：{user_query}
工具执行结果：{tool_result}

请基于工具执行结果，用自然专业的语言回复用户，解释工具结果的意义并给出建议。
"""
                                tool_response = client.models.generate_content(
                                    model='gemini-2.0-flash-exp',
                                    contents=tool_prompt,
                                )
                                return tool_response.text
                
                # 如果没有工具调用，返回文本响应
                if hasattr(candidate.content, 'parts'):
                    text_parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                    if text_parts:
                        return "".join(text_parts)
        
        # 如果上述方法都不行，使用简单方法
        return str(response.text) if hasattr(response, 'text') else "我暂时无法处理这个请求，请稍后重试。"
        
    except Exception as e:
        return f"抱歉，我在处理您的请求时遇到了技术问题。错误信息: {str(e)}"

# --- 4. 主界面 ---
def main():
    st.set_page_config(layout="wide", page_title="船舶故障电弧智能监测与预警平台")
    st.title("🚢 船舶电气安全智能监测平台")
    st.markdown("**智能诊断 · 主动预警 · 专家指导**")
    
    # 初始化状态
    if 'messages' not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant", 
                "content": "您好！我是海安，您的船舶电气安全智能助手。我可以帮您监测电力系统状态、诊断故障风险并提供维护建议。请告诉我您关心什么问题？"
            }
        ]
    if 'fault_scenario' not in st.session_state:
        st.session_state.fault_scenario = "normal"
    if 'circuit_id' not in st.session_state:
        st.session_state.circuit_id = "03号舱回路"

    # 检查密钥
    get_gemini_client()

    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 系统配置")
        st.session_state.circuit_id = st.selectbox(
            "监测回路",
            ["03号舱回路", "机舱主配电板", "货舱泵回路", "导航设备供电", "生活区供电"]
        )
        
        st.subheader("🔧 运行模式")
        scenario = st.radio(
            "选择场景:",
            ["正常运行", "早期电弧预警", "严重电弧故障", "电机启动干扰"],
            index=0
        )
        
        scenario_map = {
            "正常运行": "normal",
            "早期电弧预警": "early_arc", 
            "严重电弧故障": "severe_arc",
            "电机启动干扰": "motor_start"
        }
        st.session_state.fault_scenario = scenario_map[scenario]
        
        st.subheader("💡 对话提示")
        st.info("""
        您可以问我：
        - 当前系统状态如何？
        - 有没有潜在风险？
        - 这个故障该怎么处理？
        - 生成维护工单
        - 检查系统稳定性
        """)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.header("📊 实时监测面板")
        
        # 生成实时数据
        t_series, current_data = simulate_current_data(
            t=4000, 
            fault_scenario=st.session_state.fault_scenario,
            prediction_mode=(st.session_state.fault_scenario == "early_arc")
        )
        
        # 模型推理
        status_text, confidence, fault_type = dl_model_inference(
            current_data, st.session_state.fault_scenario
        )
        
        # 系统状态
        system_status = {
            "detection_status": status_text,
            "confidence": confidence,
            "fault_type": fault_type,
            "circuit_id": st.session_state.circuit_id,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        
        # 状态显示
        status_color = {
            "运行正常": "🟢",
            "干扰信号": "🔵", 
            "一级预警": "🟠",
            "二级预警": "🔴"
        }
        
        emoji = "🟢"
        for key, value in status_color.items():
            if key in status_text:
                emoji = value
                break
                
        st.markdown(f"### {emoji} {status_text}")
        
        col1a, col1b, col1c = st.columns(3)
        with col1a:
            st.metric("置信度", f"{confidence:.1f}%")
        with col1b:
            st.metric("监测回路", st.session_state.circuit_id)
        with col1c:
            st.metric("更新时间", system_status['timestamp'])
        
        # 波形图
        fig, ax = plt.subplots(figsize=(10, 4))
        color = 'green' if '正常' in status_text else 'red' if '二级' in status_text else 'orange'
        ax.plot(t_series, current_data, label='电流波形', color=color, linewidth=1)
        ax.set_title(f"实时电流监测 - {st.session_state.circuit_id}")
        ax.set_xlabel("时间 (ms)")
        ax.set_ylabel("电流 (A)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(-20, 20)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.header("💬 智能对话助手")
        
        # 显示对话历史
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 快捷提问按钮
        st.subheader("🚀 快捷提问")
        quick_questions = [
            "当前系统运行状态如何？",
            "检测到的是什么类型的故障？", 
            "请生成详细的诊断报告",
            "系统稳定性怎么样？",
            "根据当前情况创建维护工单",
            "这个风险需要立即处理吗？"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(quick_questions):
            with cols[i % 2]:
                if st.button(question, key=f"quick_{i}", use_container_width=True):
                    # 处理快捷提问
                    st.session_state.messages.append({"role": "user", "content": question})
                    with st.chat_message("user"):
                        st.markdown(question)

                    with st.chat_message("assistant"):
                        with st.spinner("海安正在思考..."):
                            response = gemini_agent_response(
                                question, 
                                system_status, 
                                st.session_state.messages
                            )
                        
                        # 流畅输出效果
                        message_placeholder = st.empty()
                        full_response = ""
                        for char in response:
                            full_response += char
                            message_placeholder.markdown(full_response + "▌")
                            time.sleep(0.01)
                        message_placeholder.markdown(full_response)
                    
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.rerun()
        
        # 聊天输入
        if prompt := st.chat_input("请输入您的问题..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("海安正在分析..."):
                    response = gemini_agent_response(
                        prompt, 
                        system_status, 
                        st.session_state.messages
                    )
                
                message_placeholder = st.empty()
                full_response = ""
                for char in response:
                    full_response += char
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.005)
                message_placeholder.markdown(full_response)
                
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

if __name__ == "__main__":
    main()
