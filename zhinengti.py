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
    fault_id: str = Field(description="当前故障事件的唯一标识ID，例如：'EVENT-20251028-001'")
    severity: str = Field(description="故障的严重程度，例如：'一级预警'或'二级预警'")
    fault_type: str = Field(description="故障类型，如：'串联电弧故障'、'绝缘老化'等")

class StabilityInput(BaseModel):
    """用于查询船端边缘计算单元和船岸协同通信链路的实时状态和负载率"""

class MaintenanceInput(BaseModel):
    """根据故障类型生成维护工单"""
    circuit_id: str = Field(description="回路编号，例如：'03号舱回路'")
    fault_severity: str = Field(description="故障严重程度")
    maintenance_type: str = Field(description="维护类型：预防性/紧急")

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
        "maintenance_advice": "立即进行预防性检查，紧固连接件，参考CCS规范第5.4.1条",
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
        "safety_requirements": "断电操作，穿戴PPE"
    }
    return f"【维护工单】{json.dumps(order_data, ensure_ascii=False)}"

AVAILABLE_TOOLS = {
    "generate_diagnostic_report": generate_diagnostic_report,
    "check_system_stability": check_system_stability,
    "generate_maintenance_order": generate_maintenance_order,
}

# --- 1. 增强的数据模拟 ---
def simulate_current_data(t, fault_scenario="normal", prediction_mode=False):
    """
    模拟更真实的船舶电流数据
    fault_scenario: 'normal', 'early_arc', 'severe_arc', 'motor_start'
    """
    base_frequency = 50
    time_series = np.linspace(0, 2 / base_frequency, t)  # 2个周期
    current = 10 * np.sin(2 * np.pi * base_frequency * time_series)
    
    # 基础噪声
    current += np.random.normal(0, 0.05, t)
    
    if fault_scenario == "early_arc":
        # 早期电弧特征：间歇性高频噪声
        mask = (time_series % 0.1 < 0.02)  # 10%时间出现电弧
        high_freq = np.sin(2 * np.pi * 5000 * time_series) * 0.3
        current += high_freq * mask
        
    elif fault_scenario == "severe_arc":
        # 严重电弧特征：持续高频噪声+幅值变化
        high_freq = np.sin(2 * np.pi * 3000 * time_series) * 0.8
        current += high_freq + 2 * np.random.rand(t)
        
    elif fault_scenario == "motor_start":
        # 电机启动干扰
        startup_effect = 3 * np.exp(-time_series * 2) * np.sin(2 * np.pi * 100 * time_series)
        current += startup_effect

    if prediction_mode:
        # 预测模式下的趋势特征
        trend = 0.5 * np.exp(-time_series * 3) * np.sin(2 * np.pi * 150 * time_series)
        current += trend

    return time_series * 1000, current

# --- 2. 增强的模型推理模拟 ---
def dl_model_inference(data, fault_scenario):
    """模拟双重深度学习引擎的推理结果"""
    
    # 1D-DSTN/1D-DITN 检测结果
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
    """安全地获取 Gemini 客户端"""
    try:
        GEMINI_API_KEY = st.secrets["gemini_api_key"]
        return genai.Client(api_key=GEMINI_API_KEY)
    except KeyError:
        st.error("初始化失败：无法找到 Gemini API 密钥。请在 Streamlit Cloud 的 Secrets 中配置 'gemini_api_key'。")
        st.stop()
    except Exception as e:
        st.error(f"初始化 Gemini 客户端失败: {e}")
        st.stop()

def gemini_agent_response(user_query: str, system_status: dict):
    """增强的智能体响应函数"""
    client = get_gemini_client()
    
    # 构建系统状态上下文
    status_context = (
        f"【实时系统状态】\n"
        f"- 检测状态: {system_status['detection_status']}\n"
        f"- 置信度: {system_status['confidence']}%\n" 
        f"- 故障类型: {system_status['fault_type']}\n"
        f"- 回路编号: {system_status['circuit_id']}\n"
        f"- 时间戳: {system_status['timestamp']}\n"
    )
    
    GROUNDING_FACTS = (
        "【RAG检索结果：船舶电气安全知识库精要】\n"
        "--- 1. 预测与预警（基于 Informer 模型）---\n"
        " - **一级预警特征**：电流波形呈现不规则高频震荡（1-5kHz），幅值变化±15%，这是早期电弧的明确信号。\n"
        " - **二级预警特征**：持续高频噪声（3-8kHz），电流幅值异常波动超过±30%，需立即处理。\n"
        "--- 2. 故障诊断（历史经验归因）---\n"
        " - **根本原因**：80%的船舶电弧故障源于高振动区域的电缆连接点接触不良。\n"
        " - **典型位置**：机舱、货舱泵区、甲板机械供电回路。\n"
        "--- 3. 维护规范（船级社要求）---\n"
        " - **CCS规范第5.4.1条**：高振动区域每季度必须进行预防性检查和紧固维护。\n"
        " - **ABS规范第4-8-3条**：检测到电弧故障后，需在24小时内完成根本原因分析。\n"
    )

    system_instruction = (
        "你是一个专业的船舶电气安全智能体，基于船岸协同架构工作。"
        "你必须结合实时系统状态、RAG知识库和可用工具来提供准确的诊断和建议。"
        "对于预警信息，要明确说明风险等级和应对措施；对于故障诊断，要引用相关规范条款。"
        f"当前系统状态：{status_context}"
    )
    
    full_prompt = system_instruction + "\n\n" + GROUNDING_FACTS + "\n\n用户提问：" + user_query

    try:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=list(AVAILABLE_TOOLS.values()),
        )
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
            config=config,
        )
        
        if response.function_calls:
            for function_call in response.function_calls:
                tool_name = function_call.name
                tool_args = dict(function_call.args)
                
                if tool_name in AVAILABLE_TOOLS:
                    tool_result = AVAILABLE_TOOLS[tool_name](**tool_args)
                    
                    response_after_tool = client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[
                            types.Content(role="user", parts=[types.Part.from_text(full_prompt)]),
                            types.Content(role="model", parts=[types.Part.from_function_call(function_call)]),
                            types.Content(role="tool", parts=[types.Part.from_text(tool_result)]),
                        ],
                        config=types.GenerateContentConfig(system_instruction=system_instruction),
                    )
                    return response_after_tool.text 
                
        return response.text

    except Exception as e:
        return f"智能体 API 调用失败。错误信息: {e}"

# --- 4. 主界面 ---
def main():
    st.set_page_config(layout="wide", page_title="船舶故障电弧智能监测与预警平台")
    st.title("🚢 船舶故障电弧智能监测与预警平台")
    st.markdown("**船岸协同架构 | 双重深度学习引擎 | 大模型智能体赋能**")

    # 初始化状态
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'fault_scenario' not in st.session_state:
        st.session_state.fault_scenario = "normal"
    if 'circuit_id' not in st.session_state:
        st.session_state.circuit_id = "03号舱回路"
    if 'last_update' not in st.session_state:
        st.session_state.last_update = time.time()

    # 检查密钥
    get_gemini_client()

    # 侧边栏 - 系统配置
    with st.sidebar:
        st.header("系统配置")
        st.session_state.circuit_id = st.selectbox(
            "监测回路",
            ["03号舱回路", "机舱主配电板", "货舱泵回路", "导航设备供电"]
        )
        
        st.subheader("故障场景模拟")
        scenario = st.radio(
            "选择运行模式:",
            ["正常运行", "早期电弧预警", "严重电弧故障", "电机启动干扰"]
        )
        
        scenario_map = {
            "正常运行": "normal",
            "早期电弧预警": "early_arc", 
            "严重电弧故障": "severe_arc",
            "电机启动干扰": "motor_start"
        }
        st.session_state.fault_scenario = scenario_map[scenario]
        
        # 系统信息
        st.subheader("系统信息")
        st.info("""
        **架构层级:**
        - 🚢 船端边缘计算
        - ☁️ 岸基智能体
        - 🔗 船岸协同
        """)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.header("📊 实时监测 Dashboard")
        
        # 实时数据生成
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
            "运行正常": "green",
            "干扰信号": "blue", 
            "一级预警": "orange",
            "二级预警": "red"
        }
        
        color = "green"
        for key, value in status_color.items():
            if key in status_text:
                color = value
                break
                
        st.markdown(
            f"**检测状态:** <span style='color:{color}; font-size: 24px;'>{status_text}</span>",
            unsafe_allow_html=True
        )
        st.metric("模型置信度", f"{confidence:.1f}%")
        st.metric("监测回路", st.session_state.circuit_id)
        
        # 波形图
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_series, current_data, label='电流波形 (A)', color=color, linewidth=1)
        ax.set_title(f"实时电流波形监测 - {st.session_state.circuit_id}")
        ax.set_xlabel("时间 (ms)")
        ax.set_ylabel("电流 (A)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(-20, 20)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        
        # 频谱分析（简化版）
        if "预警" in status_text:
            st.warning("🔍 检测到高频噪声成分，建议进行详细频谱分析")

    with col2:
        st.header("💬 智能体交互中心")
        
        # 显示历史消息
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 预设问题
        st.subheader("💡 预设问题")
        presets = {
            "前瞻预警": "当前波形走势是否正常？有无潜在的电弧风险？",
            "诊断查询": "请分析故障根本原因和船级社维护要求",
            "系统状态": "边缘计算单元和通信链路状态如何？",
            "维护指导": "根据当前预警生成维护工单"
        }
        
        for preset_name, preset_text in presets.items():
            if st.button(f"{preset_name}: {preset_text}", key=preset_name):
                # 保存用户消息
                st.session_state.messages.append({"role": "user", "content": preset_text})
                with st.chat_message("user"):
                    st.markdown(preset_text)

                # 生成智能体响应
                with st.chat_message("assistant"):
                    with st.spinner("智能体推理中..."):
                        response = gemini_agent_response(preset_text, system_status)
                    
                    # 模拟打字效果
                    full_response = ""
                    message_placeholder = st.empty()
                    for chunk in response.split():
                        full_response += chunk + " "
                        time.sleep(0.01)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    
                # 保存智能体响应
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()
        
        # 聊天输入
        if prompt := st.chat_input("请输入您的问题..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("智能体推理中..."):
                    response = gemini_agent_response(prompt, system_status)
                
                full_response = ""
                message_placeholder = st.empty()
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.01)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

if __name__ == "__main__":
    main()
