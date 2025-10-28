import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# --- 0. 环境和工具定义 ---

# 定义工具的输入结构
class ReportInput(BaseModel):
    """用于生成详细故障诊断报告的工具"""
    fault_id: str = Field(description="当前故障事件的唯一标识ID，例如：'EVENT-20251028-001'")
    severity: str = Field(description="故障的严重程度，例如：'一级预警'或'二级预警'")

class StabilityInput(BaseModel):
    """用于查询船端边缘计算单元和船岸协同通信链路的实时状态和负载率"""

# --- 定义工具函数 ---
# 工具函数必须返回一个字符串，作为 LLM 的上下文输入
def generate_diagnostic_report(fault_id: str = "CURRENT-FAULT", severity: str = "Level 2") -> str:
    """
    此工具用于根据DL模型结果和LLM诊断结果生成格式化的PDF故障诊断报告，并发送到运维中心。
    """
    # 模拟实际操作，返回操作结果
    return f"【工具调用结果】已自动生成并发送《{severity} 级诊断报告 ({fault_id})》至您的运维终端。"

def check_system_stability() -> str:
    """
    此工具用于查询船端边缘计算单元和船岸协同通信链路的实时状态和负载率。
    """
    # 模拟查询系统状态，返回实时结果
    # 注意：这里的返回结果依然是“硬编码”的系统数据，但这是工具获取数据的本质
    return (
        "**系统状态数据**：船端边缘计算单元负载率为38%，模型推理延迟为15ms，链路延迟低于50ms。"
    )

# 将工具函数打包
AVAILABLE_TOOLS = {
    "generate_diagnostic_report": generate_diagnostic_report,
    "check_system_stability": check_system_stability,
}


# --- 1. 数据模拟（不变）---
def simulate_current_data(t, is_fault, prediction_mode=False):
    base_frequency = 50
    time_series = np.linspace(0, 1 / base_frequency, t)
    current = 10 * np.sin(2 * np.pi * base_frequency * time_series) + np.random.normal(0, 0.1, t)

    if is_fault:
        high_freq_noise = np.sin(2 * np.pi * 5000 * time_series) * 0.5 * np.random.rand(t)
        arc_pulse = np.maximum(0, np.sin(2 * np.pi * 50 * time_series) - 0.5) * 5
        current += high_freq_noise * arc_pulse

    if prediction_mode:
        current += 0.5 * np.exp(-time_series * 5) * np.sin(2 * np.pi * 200 * time_series)

    return time_series * 1000, current


# --- 2. 模型推理模拟（不变）---
def dl_model_inference(data):
    if data.max() > 14 or data.min() < -14:
        return "二级预警 (故障确认)", 97.5
    elif data.max() > 12 or data.min() < -12:
        return "一级预警 (预测风险)", 75.0
    else:
        return "运行正常 (安全)", 5.0


# --- 3. 智能体核心逻辑（使用 Gemini API，最少硬编码）---
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

def gemini_agent_response(user_query: str, current_status: str):
    client = get_gemini_client()
    
    # *** 极简 RAG/Informer 事实上下文（LLM 必须基于这些事实进行自我生成）***
    GROUNDING_FACTS = (
        "【船舶安全核心事实】：\n"
        "1. **预测模型**：如果当前是'一级预警'，Informer网络的分析结果是：电流波形走势将呈现持续恶化的不规则高频震荡，需立即检查。\n"
        "2. **故障根本原因**：该类故障的根本原因为高振动区域的电缆固定件老化松动。\n"
        "3. **维护规范**：船级社规范[XX-2023]第5.4.1条要求：高振动区域的电气连接点应每季度进行预防性检查。\n"
    )

    system_instruction = (
        "你是一个专业的船舶电气安全智能体。你的回复必须**完全基于**你拥有的工具和提供的【船舶安全核心事实】进行推理和组织语言。"
        "请用**专业、严谨**的语气回复用户。当前模型检测状态为："
        f"【实时状态】: {current_status}。"
    )
    
    full_prompt = GROUNDING_FACTS + "\n\n用户提问：" + user_query

    try:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=list(AVAILABLE_TOOLS.values()),
        )
        
        # 第一次 API 调用，让 Gemini 决定是回复还是调用工具
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=full_prompt,
            config=config,
        )
        
        # 1. 处理 Tool-Calling
        if response.function_calls:
            for function_call in response.function_calls:
                tool_name = function_call.name
                tool_args = dict(function_call.args)
                
                if tool_name in AVAILABLE_TOOLS:
                    # 执行本地工具函数获取数据
                    tool_result = AVAILABLE_TOOLS[tool_name](**tool_args)
                    
                    # 第二次 API 调用：将工具执行结果反馈给 LLM，让其生成最终回复
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
                
        # 2. 返回 LLM 的标准文本回复（LLM 自我生成）
        return response.text

    except Exception as e:
        return f"智能体 API 调用失败。请检查密钥或网络连接。错误信息: {e}"


# --- 4. 主界面（只需要替换调用函数）---
def main():
    st.set_page_config(layout="wide", page_title="船舶故障电弧智能监测与预警平台")
    st.title("🚢 船舶故障电弧智能监测与预警平台 (演示原型)")

    # 初始化状态
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'is_fault_active' not in st.session_state:
        st.session_state.is_fault_active = False
    if 't_start' not in st.session_state:
        st.session_state.t_start = time.time()
    
    # 检查密钥和初始化客户端
    get_gemini_client() 

    col1, col2 = st.columns([3, 2])

    with col1:
        st.header("实时监测与预警 Dashboard (船端边缘计算模拟)")

        # 故障切换按钮（用回调更新状态）
        def toggle_fault():
            st.session_state.is_fault_active = not st.session_state.is_fault_active
            st.session_state.t_start = time.time()
            st.toast(f"故障状态已切换至: {'故障模式' if st.session_state.is_fault_active else '正常模式'}")
            
        st.button(
            "🔴 模拟故障电弧发生 / 🟢 恢复正常运行",
            on_click=toggle_fault
        )

        # 生成实时数据
        t_series, current_data = simulate_current_data(
            t=2000,
            is_fault=st.session_state.is_fault_active,
            prediction_mode=(time.time() - st.session_state.t_start < 20 and not st.session_state.is_fault_active)
        )
        status_text, confidence = dl_model_inference(current_data)

        # 显示状态
        color = "green"
        if "一级预警" in status_text:
            color = "orange"
        elif "二级预警" in status_text:
            color = "red"
        st.markdown(
            f"**模型检测状态:** <span style='color:{color}; font-size: 20px;'>{status_text}</span> | **置信度:** {confidence:.1f}%",
            unsafe_allow_html=True
        )

        # 绘制波形图
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_series, current_data, label='电流波形 (A)', color=color)
        ax.set_title("实时电流波形监测 (船端)")
        ax.set_xlabel("时间 (ms)")
        ax.set_ylabel("电流 (A)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(-15, 15)
        st.pyplot(fig)
        plt.close(fig)
        
        # 循环刷新逻辑 (保持不变)
        time.sleep(0.5) 
        st.rerun()


    with col2:
        st.header("智能体交互中心 (岸基运维中心模拟)")

        # 显示历史消息
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 预设对话引导
        st.subheader("💡 预设演示对话 (请手动输入)")
        st.info("1. **前瞻预警**: 请问当前波形走势是否正常？有无潜在的电弧风险？")
        st.info("2. **诊断查询**: 请问如何查询故障电弧发生的根本原因，以及船级社的维护要求？")
        st.info("3. **系统状态**: 请告知我边缘计算单元的负载率和系统稳定性如何？")

        # 聊天输入处理
        if prompt := st.chat_input("请输入您的问题..."):
            # 保存用户消息
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 生成智能体响应
            with st.chat_message("assistant"):
                current_status = status_text
                
                # *** 调用 Gemini API 进行生成 ***
                with st.spinner("智能体正在进行诊断和推理..."):
                     response = gemini_agent_response(prompt, current_status)
                
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
            # 强制刷新以显示最新的聊天记录和按钮状态
            st.rerun() 


if __name__ == "__main__":
    main()
