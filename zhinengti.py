import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time


# --- 1. 数据模拟 ---
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


# --- 2. 模型推理模拟 ---
def dl_model_inference(data):
    if data.max() > 14 or data.min() < -14:
        return "二级预警 (故障确认)", 97.5
    elif data.max() > 12 or data.min() < -12:
        return "一级预警 (预测风险)", 75.0
    else:
        return "运行正常 (安全)", 5.0


# --- 3. 智能体逻辑 ---
def intelligent_agent_response(user_query, current_status):
    query = user_query.lower()
    if any(k in query for k in ["波形走势", "预测", "潜在风险"]):
        if "一级预警" in current_status:
            return (
                "**[时序预测结果]**：经基于**Informer网络**的模型分析，预测电流波形走势将呈现**持续恶化的不规则高频震荡**，符合一级（早期）串联故障电弧的起始特征。\n\n"
                "**[处置建议]**：请船员立即检查该回路负载端口的温度异常，并准备预防性维护工单，等待进一步指示。已同步岸基运维中心。"
            )
        else:
            return "当前波形走势稳定，未检测到早期故障电弧的预测特征，系统运行平稳。"
    elif any(k in query for k in ["规范", "维护要求", "根本原因"]):
        return (
            "**[诊断分析]**：历史数据显示，该类故障主要原因为高振动区域的**电缆固定件老化松动**，导致接头阻抗增加并产生间歇性放电。\n\n"
            "**[规范查询]**：根据**[RAG知识库]**，船级社规范[XX-2023]第5.4.1条要求：对于高振动区域的电气连接点，应每季度进行预防性检查。\n\n"
            "**[工具调用]**：已自动生成并发送**《高风险部件检查指导手册》**至您的工作终端。"
        )
    elif any(k in query for k in ["稳定性", "负载率", "系统状态"]):
        return (
            "**[系统状态]**：船端边缘计算单元当前负载率稳定在38%，模型推理延迟在15ms以内，**稳定性良好**。船岸协同通信链路延迟低于50ms，状态稳定。"
        )
    else:
        return f"您好，我是船舶安全智能体，当前系统状态为：**{current_status}**。请问您需要进行故障诊断、安全规范查询，还是系统状态检查？"


# --- 4. 主界面 ---
def main():
    st.set_page_config(layout="wide", page_title="船舶故障电弧智能监测与预警平台")
    st.title("🚢 船舶故障电弧智能监测与预警平台 (演示原型)")

    # 初始化状态（确保聊天记录不会因刷新丢失）
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'is_fault_active' not in st.session_state:
        st.session_state.is_fault_active = False
    if 't_start' not in st.session_state:
        st.session_state.t_start = time.time()
    # 新增：控制刷新的标记
    if 'refresh_counter' not in st.session_state:
        st.session_state.refresh_counter = 0

    col1, col2 = st.columns([3, 2])

    with col1:
        st.header("实时监测与预警 Dashboard (船端边缘计算模拟)")

        # 故障切换按钮（用回调更新状态）
        def toggle_fault():
            st.session_state.is_fault_active = not st.session_state.is_fault_active
            st.session_state.t_start = time.time()
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

    with col2:
        st.header("智能体交互中心 (岸基运维中心模拟)")

        # 显示历史消息（从session_state读取，确保刷新后不丢失）
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
                response = intelligent_agent_response(prompt, current_status)
                # 模拟打字效果
                full_response = ""
                message_placeholder = st.empty()
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.02)
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            # 保存智能体响应
            st.session_state.messages.append({"role": "assistant", "content": response})



if __name__ == "__main__":
    main()
