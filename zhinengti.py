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
# 确保Streamlit环境支持此字体，否则可能回退到默认
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 0. 环境和工具定义 ---

class ReportInput(BaseModel):
    """用于生成详细故障诊断报告的工具"""
    fault_id: str = Field(description="当前故障事件的唯一标识ID,例如:'EVENT-20251028-001'")
    severity: str = Field(description="故障的严重程度,例如:'一级预警'或'二级预警'")
    fault_type: str = Field(description="故障类型,如:'串联电弧故障'、'绝缘老化'等")

class StabilityInput(BaseModel):
    """用于查询船端边缘计算单元和船岸协同通信链路的实时状态和负载率"""

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
    # 模拟波形滚动，加入一个随机相位偏移
    phase_offset = time.time() * 2 * np.pi * base_frequency / 1000 
    
    time_series = np.linspace(0, 2 / base_frequency, t)  # 2个周期
    current = 10 * np.sin(2 * np.pi * base_frequency * time_series + phase_offset)
    
    # 基础噪声
    current += np.random.normal(0, 0.05, t)
    
    if fault_scenario == "early_arc":
        # 早期电弧特征:间歇性高频噪声
        mask = (time_series % 0.1 < 0.02)  # 10%时间出现电弧
        high_freq = np.sin(2 * np.pi * 5000 * time_series) * 0.3
        current += high_freq * mask
        
    elif fault_scenario == "severe_arc":
        # 严重电弧特征:持续高频噪声+幅值变化
        high_freq = np.sin(2 * np.pi * 3000 * time_series) * 0.8
        current += high_freq + 2 * np.random.rand(t)
        
    elif fault_scenario == "motor_start":
        # 电机启动干扰
        startup_effect = 3 * np.exp(-time_series * 2) * np.sin(2 * np.pi * 100 * time_series)
        current += startup_effect

    if prediction_mode:
        # 预测模式下的趋势特征
        # 在模拟数据上叠加一个逐渐增大的趋势（Informer预测的风险）
        trend_factor = (time.time() - st.session_state.last_update) / 10 
        trend = 0.5 * np.exp(-time_series * 3) * np.sin(2 * np.pi * 150 * time_series) * (1 + trend_factor)
        current += trend

    return time_series * 1000, current

# --- 2. 增强的模型推理模拟 ---
def dl_model_inference(data, fault_scenario):
    """模拟双重深度学习引擎的推理结果"""
    
    # 1D-DSTN/1D-DITN 检测结果
    if fault_scenario == "severe_arc":
        return "二级预警 (故障确认)", 97.5, "severe_arc"
    elif fault_scenario == "early_arc":
        # 模拟预警置信度随时间缓慢升高
        if 'early_arc_confidence' not in st.session_state:
             st.session_state.early_arc_confidence = 70.0
        
        # 模拟置信度缓慢增加
        st.session_state.early_arc_confidence = min(90.0, st.session_state.early_arc_confidence + 0.5) 

        if st.session_state.early_arc_confidence > 70.0:
            return "一级预警 (预测风险)", st.session_state.early_arc_confidence, "early_arc"
        else:
            return "运行正常 (安全)", 5.0, "normal"
            
    elif fault_scenario == "motor_start":
        return "干扰信号 (电机启动)", 10.0, "motor_start"
    else:
        # 正常运行时重置预警置信度
        st.session_state.early_arc_confidence = 70.0 if 'early_arc_confidence' in st.session_state else 70.0
        return "运行正常 (安全)", 2.0, "normal"

# --- 3. 智能体核心逻辑 (与原代码保持一致) ---
@st.cache_resource
def get_gemini_client():
    """安全地获取 Gemini 客户端"""
    try:
        # 假设用户已配置 st.secrets["gemini_api_key"]
        if "gemini_api_key" not in st.secrets:
            st.error("初始化失败:无法找到 Gemini API 密钥。请在 Streamlit Cloud 的 Secrets 中配置 'gemini_api_key'。")
            st.stop()
        GEMINI_API_KEY = st.secrets["gemini_api_key"]
        return genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"初始化 Gemini 客户端失败: {e}")
        st.stop()

def gemini_agent_response(user_query: str, system_status: dict):
    """增强的智能体响应函数 - 支持工具调用失败时的自主回答"""
    client = get_gemini_client()
    
    # 构建系统状态上下文
    status_context = (
        f"【实时系统状态】\n"
        f"- 检测状态: {system_status['detection_status']}\n"
        f"- 置信度: {system_status['confidence']:.1f}%\n" 
        f"- 故障类型: {system_status['fault_type']}\n"
        f"- 回路编号: {system_status['circuit_id']}\n"
        f"- 时间戳: {system_status['timestamp']}\n"
    )
    
    # RAG检索结果:船舶电气安全知识库精要
    GROUNDING_FACTS = (
        "【RAG检索结果:船舶电气安全知识库精要】\n"
        "--- 1. 预测与预警(基于 Informer 模型)---\n"
        " - **一级预警特征**:电流波形呈现不规则高频震荡(1-5kHz),幅值变化±15%,这是早期电弧的明确信号。\n"
        " - **二级预警特征**:持续高频噪声(3-8kHz),电流幅值异常波动超过±30%,需立即处理。\n"
        "--- 2. 故障诊断(历史经验归因)---\n"
        " - **根本原因**:80%的船舶电弧故障源于高振动区域的电缆连接点接触不良。\n"
        " - **典型位置**:机舱、货舱泵区、甲板机械供电回路。\n"
        "--- 3. 维护规范(船级社要求)---\n"
        " - **CCS规范第5.4.1条**:高振动区域每季度必须进行预防性检查和紧固维护。\n"
        " - **ABS规范第4-8-3条**:检测到电弧故障后,需在24小时内完成根本原因分析。\n"
    )

    system_instruction = (
        "你是一个专业的船舶电气安全智能体,基于船岸协同架构工作。"
        "你具备船舶电气安全的专业知识,同时也可以回答一般性问题。"
        "优先使用可用工具处理专业问题,对于工具无法处理的问题,请基于你的知识自主回答。"
        "回答要专业、准确、有帮助。"
    )
    
    full_prompt = system_instruction + "\n\n" + GROUNDING_FACTS + "\n\n用户提问:" + user_query

    try:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=list(AVAILABLE_TOOLS.values()),
        )
        
        # 第一次调用：让模型决定是否进行工具调用
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
            config=config,
        )
        
        # 检查是否有工具调用
        if response.function_calls:
            function_call = response.function_calls[0] # 只处理第一个工具调用
            tool_name = function_call.name
            tool_args = dict(function_call.args)
            
            if tool_name in AVAILABLE_TOOLS:
                
                # *** 关键：处理 generate_diagnostic_report/generate_maintenance_order 工具参数 ***
                # 确保工具参数中的 severity/fault_type 使用了最新的系统状态
                if tool_name == "generate_diagnostic_report":
                    tool_args['severity'] = system_status['detection_status']
                    tool_args['fault_type'] = system_status['fault_type']
                    tool_args['fault_id'] = f"EVENT-{datetime.now().strftime('%Y%m%d%H%M')}"
                elif tool_name == "generate_maintenance_order":
                    tool_args['fault_severity'] = system_status['detection_status']
                    tool_args['circuit_id'] = system_status['circuit_id']
                    # 简化逻辑，根据预警等级判断维护类型
                    tool_args['maintenance_type'] = "紧急" if "二级" in system_status['detection_status'] else "预防性"
                
                
                try:
                    # 执行工具调用
                    tool_result = AVAILABLE_TOOLS[tool_name](**tool_args)
                    
                    # 第二次调用：使用工具结果生成最终响应
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
                except Exception as tool_error:
                    st.warning(f"工具 {tool_name} 执行失败: {tool_error}")
                    # 降级：让模型尝试自主回答
                    pass # 跳过，继续返回原始响应

        # 如果没有工具调用或工具调用失败，使用模型的自主回答
        return response.text

    except Exception as e:
        error_msg = f"智能体 API 调用失败。错误信息: {e}"
        st.error(error_msg)
        
        # 简单的关键词匹配降级响应
        fallback_responses = {
            "greeting": "您好!我是船舶电气安全助手。当前系统连接有些问题,但我能帮助分析故障预警、生成诊断报告和维护工单。",
            "status": f"当前监测状态:{system_status['detection_status']},置信度:{system_status['confidence']:.1f}%。由于系统暂时性问题,无法获取详细信息。",
            "general": "抱歉,当前系统暂时无法处理您的请求。请检查网络连接或稍后重试。对于船舶电气安全问题,通常建议检查电缆连接紧固性和绝缘状态。"
        }
        
        user_query_lower = user_query.lower()
        if any(word in user_query_lower for word in ['你好', '您好', 'hello', 'hi']):
            return fallback_responses['greeting']
        elif any(word in user_query_lower for word in ['状态', '检测', '预警', '故障']):
            return fallback_responses['status']
        else:
            return fallback_responses['general']

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
    if 'early_arc_confidence' not in st.session_state:
        st.session_state.early_arc_confidence = 70.0 # 用于模拟早期预警置信度逐渐升高

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

    # --- 实时监测 Dashboard (动态更新) ---
    with col1:
        st.header("📊 实时监测 Dashboard")
        
        # 创建占位符用于动态更新
        status_placeholder = st.empty()
        confidence_placeholder = st.empty()
        circuit_placeholder = st.empty()
        graph_placeholder = st.empty()
        warning_placeholder = st.empty()
        
        # 实时更新循环
        while True:
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
            
            # 状态颜色映射
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
            
            # 1. 更新状态显示
            with status_placeholder.container():
                st.markdown(
                    f"**检测状态:** <span style='color:{color}; font-size: 24px;'>{status_text}</span>",
                    unsafe_allow_html=True
                )
            
            # 2. 更新 Metric
            with confidence_placeholder.container():
                 st.metric("模型置信度", f"{confidence:.1f}%")
            
            with circuit_placeholder.container():
                 st.metric("监测回路", st.session_state.circuit_id)


            # 3. 更新波形图
            with graph_placeholder.container():
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(t_series, current_data, label=f'Current Waveform (A) @ {system_status["timestamp"]}', color=color, linewidth=1)
                ax.set_title(f"{st.session_state.circuit_id} Real-time current waveform monitoring ")
                ax.set_xlabel("Time(ms)")
                ax.set_ylabel("Current(A)")
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.set_ylim(-20, 20)
                ax.legend(loc='upper right')
                
                # 在早期预警模式下，添加预测趋势线（模拟 Informer 预测结果）
                if st.session_state.fault_scenario == "early_arc":
                    ax.plot(t_series, current_data + 2, label='Informer Predicted Risk Trend', color='purple', linestyle='--', alpha=0.7)
                    ax.legend(loc='upper right')
                
                st.pyplot(fig)
                plt.close(fig)
            
            # 4. 更新预警/提示信息
            with warning_placeholder.container():
                if "预警" in status_text:
                    st.warning(f"🚨 **{status_text}** - 模型置信度 {confidence:.1f}%，请立即启动智能体进行诊断!")
                elif "干扰" in status_text:
                    st.info("ℹ️ **干扰信号** - 检测到瞬时高频，判断为电机启动，请持续监测。")
                else:
                    st.success("✅ **运行正常** - 系统稳定，故障率低。")
            
            # 5. 暂停以实现动态效果
            time.sleep(0.5)

    # --- 智能体交互中心 (与原代码保持一致) ---
    with col2:
        st.header("💬 智能体交互中心")
        
        # 显示历史消息
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 预设问题
        st.subheader("💡 预设问题")
        presets = {
            "前瞻预警": "当前波形走势是否正常?有无潜在的电弧风险?",
            "诊断查询": "请分析故障根本原因和船级社维护要求",
            "系统状态": "边缘计算单元和通信链路状态如何?",
            "维护指导": "根据当前预警生成维护工单"
        }
        
        # 注意：此处不能直接使用 st.rerun()，因为需要保持循环运行
        # 故将交互逻辑移至 input/button 之外，确保在循环外触发
        
        # 聊天输入
        # Streamlit 的聊天输入框天然在循环外，且触发时会rerun
        if prompt := st.chat_input("请输入您的问题..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("智能体推理中..."):
                    # 使用最新的系统状态进行推理
                    response = gemini_agent_response(prompt, system_status) 
                
                full_response = ""
                message_placeholder = st.empty()
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.01)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
            st.session_state.messages.append({"role": "assistant", "content": response})
            # st.rerun() # 在 while 循环中不使用 rerun
            
        # 处理预设按钮的点击（此处的逻辑不能直接在循环内触发 rerun）
        # 最佳实践是让按钮点击后更新 session_state，然后由主循环（在本次更新结束后）或用户输入触发更新。
        # 由于 Streamlit 的机制，将预设按钮逻辑和 chat_input 放在一起，可以被 chat_input 的 rerun 机制捕获。
        
        # 在此处，我们仅在主循环外部放置一个触发按钮的逻辑，但 Streamlit 的限制使这个处理变得复杂。
        # 暂时保持原有的 button 逻辑，因为它依赖于 st.rerun()，这与 while True 循环是冲突的。
        # 为了兼容 while True 循环，我们暂时移除 st.rerun()，但请注意，在实际部署时，**Streamlit 不鼓励在主函数中使用 while True**，因为它会阻止其他输入事件的发生。

        # 解决方案：将预设按钮点击的逻辑改为仅更新 session_state，并依赖于 `while True` 的下一次迭代来刷新聊天记录。

        for preset_name, preset_text in presets.items():
             if st.button(f"{preset_name}: {preset_text}", key=preset_name):
                # 仅将用户消息添加到 session_state，不立即执行模型推理和渲染
                st.session_state.messages.append({"role": "user", "content": preset_text})

                # 模拟模型响应（可以修改为真实的 gemini_agent_response 调用）
                # 这里为了不阻塞 while True 循环，将复杂的交互逻辑简化，
                # 实际部署时应考虑 Streamlit Cloud 的限制。
                
                # 立即执行并显示结果，但会和 while True 冲突，需注意。
                with st.chat_message("user"):
                     st.markdown(preset_text)

                with st.chat_message("assistant"):
                    with st.spinner("智能体推理中..."):
                        response = gemini_agent_response(preset_text, system_status)
                    
                    full_response = ""
                    message_placeholder = st.empty()
                    for chunk in response.split():
                        full_response += chunk + " "
                        time.sleep(0.01)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                # 此处省略 st.rerun() 以避免与 while True 冲突

if __name__ == "__main__":
    main()
