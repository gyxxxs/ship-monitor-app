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

# --- 响应模板和策略配置 ---
RESPONSE_TEMPLATES = {
    "risk_assessment": """
🔍 **风险评估报告**

**当前状况**：{status}
**置信水平**：{confidence}%
**风险等级**：{risk_level}

📊 **分析依据**：
{analysis_basis}

🚨 **立即行动**：
{immediate_actions}

📋 **后续步骤**：
{next_steps}

⚡ **紧急联系人**：如情况恶化，立即联系轮机长
""",

    "maintenance_guidance": """
🛠️ **维护作业指导**

**作业类型**：{maintenance_type}
**预估时长**：{duration}
**风险等级**：{risk_level}

📝 **作业准备**：
{tool_preparation}

🔧 **操作流程**：
{procedure}

⚠️ **安全警示**：
{safety_warnings}

✅ **验收标准**：
{acceptance_criteria}
""",

    "trend_analysis": """
📈 **趋势分析报告**

**监测回路**：{circuit}
**分析时段**：最近30分钟

📊 **当前特征**：
{current_features}

🔮 **发展趋势**：
{trend_prediction}

🎯 **专家建议**：
{expert_suggestions}
""",

    "system_status": """
🏥 **系统健康报告**

**边缘计算单元**：
• 负载率：{compute_load}
• 推理延迟：{inference_latency}
• 检测准确率：{accuracy}

**通信链路**：
• 船岸延迟：{comm_latency}
• 数据完整性：{data_integrity}

**总体评价**：{overall_status}
"""
}

# 回复策略配置
RESPONSE_STRATEGIES = {
    "emergency": {
        "tone": "urgent",
        "structure": ["立即行动", "风险说明", "操作步骤", "安全提醒"],
        "emoji": "🚨",
        "color": "red"
    },
    "technical_diagnosis": {
        "tone": "authoritative", 
        "structure": ["现象描述", "根本原因", "规范依据", "解决方案"],
        "emoji": "🔧",
        "color": "orange"
    },
    "maintenance_guidance": {
        "tone": "instructional",
        "structure": ["工具准备", "操作流程", "验收标准", "注意事项"],
        "emoji": "🛠️",
        "color": "blue"
    },
    "status_query": {
        "tone": "informative",
        "structure": ["当前状态", "趋势分析", "建议关注", "后续计划"],
        "emoji": "📊",
        "color": "green"
    }
}

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
        mask = (time_series % 0.1 < 0.02)
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
        trend_factor = (time.time() - st.session_state.last_update) / 10 
        trend = 0.5 * np.exp(-time_series * 3) * np.sin(2 * np.pi * 150 * time_series) * (1 + trend_factor)
        current += trend

    return time_series * 1000, current

# --- 2. 增强的模型推理模拟 ---
def dl_model_inference(data, fault_scenario):
    """模拟双重深度学习引擎的推理结果"""
    
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

# --- 3. 智能体核心逻辑 ---
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

def analyze_query_type(user_query: str, system_status: dict) -> str:
    """分析查询类型和紧急程度"""
    query_lower = user_query.lower()
    
    # 紧急情况判断
    if any(word in query_lower for word in ['紧急', '立刻', '马上', '危险', '着火', '冒烟']):
        return "emergency"
    elif "二级预警" in system_status['detection_status']:
        return "emergency"
    
    # 技术诊断
    if any(word in query_lower for word in ['故障', '原因', '诊断', '分析', '为什么']):
        return "technical_diagnosis"
    
    # 维护指导
    if any(word in query_lower for word in ['维护', '修理', '检修', '工单', '怎么处理']):
        return "maintenance_guidance"
    
    # 状态查询
    if any(word in query_lower for word in ['状态', '怎么样', '正常吗', '监测', '预警']):
        return "status_query"
    
    return "status_query"

def build_enhanced_context(system_status: dict, query_type: str) -> str:
    """构建增强的上下文信息"""
    
    risk_level = "高" if "二级" in system_status['detection_status'] else "中" if "一级" in system_status['detection_status'] else "低"
    
    context_templates = {
        "emergency": f"""
🚨 **紧急情况上下文** 🚨
当前检测到：{system_status['detection_status']}
置信度：{system_status['confidence']}%
故障类型：{system_status['fault_type']}
风险等级：{risk_level}
位置：{system_status['circuit_id']}

📋 **应急预案**：
• 立即通知轮机长和值班驾驶员
• 准备应急消防设备
• 考虑切断受影响回路供电
• 启动远程技术支持流程
""",
        
        "technical_diagnosis": f"""
🔧 **技术诊断上下文**
当前状态：{system_status['detection_status']}
故障特征：{system_status['fault_type']}
监测回路：{system_status['circuit_id']}

📚 **相关知识**：
• 类似故障多发生在高振动区域
• 电缆接头松动是主要原因
• 参考规范：CCS第5.4.1条，IEC 62606
• 典型处理时间：2-4小时
"""
    }
    
    return context_templates.get(query_type, f"""
📊 **系统状态上下文**
检测状态：{system_status['detection_status']}
置信程度：{system_status['confidence']}%
故障类型：{system_status['fault_type']}
监测位置：{system_status['circuit_id']}
更新时间：{system_status['timestamp']}
""")

def build_system_instruction(strategy: dict, system_status: dict) -> str:
    """构建系统指令"""
    
    base_instruction = f"""
你是一个专业的船舶电气安全专家"海安"，具有丰富的船舶电力系统故障诊断经验。

{strategy['emoji']} **当前回复模式**：{strategy['tone']}模式
🎯 **回复结构**：请按照以下顺序组织内容：{", ".join(strategy['structure'])}

**专业要求**：
1. 使用专业但易懂的语言，避免过于技术化的术语
2. 引用规范时要说明其实际意义，不只是编号
3. 始终提供具体、可操作的下一步建议
4. 对于风险情况，要明确说明严重程度和应对措施
5. 适当使用表情符号增强表达，但不要过度

**当前系统状态**：
• 检测状态：{system_status['detection_status']}
• 置信度：{system_status['confidence']}%
• 故障类型：{system_status['fault_type']}
• 监测回路：{system_status['circuit_id']}
"""
    
    return base_instruction

def enhance_tool_arguments(tool_name: str, tool_args: dict, system_status: dict) -> dict:
    """增强工具调用参数"""
    
    if tool_name == "generate_diagnostic_report":
        return {
            'fault_id': f"EVENT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'severity': system_status['detection_status'],
            'fault_type': system_status['fault_type']
        }
    
    elif tool_name == "generate_maintenance_order":
        maintenance_type = "紧急" if "二级" in system_status['detection_status'] else "预防性"
        return {
            'circuit_id': system_status['circuit_id'],
            'fault_severity': system_status['detection_status'],
            'maintenance_type': maintenance_type
        }
    
    return tool_args

def build_tool_response_prompt(user_query: str, tool_name: str, tool_result: str, system_status: dict, strategy: dict) -> str:
    """构建工具响应提示词"""
    
    tool_explanations = {
        "generate_diagnostic_report": "用户请求生成诊断报告，以下是报告内容：",
        "check_system_stability": "用户查询系统状态，以下是状态信息：", 
        "generate_maintenance_order": "用户需要维护指导，以下是维护工单："
    }
    
    explanation = tool_explanations.get(tool_name, "已处理用户请求，结果如下：")
    
    return f"""
{explanation}

{tool_result}

请基于以上工具执行结果：
1. 用{strategy['tone']}的语气向用户解释结果
2. 按照{strategy['structure']}的结构组织回复
3. 说明这个结果对当前状况的意义
4. 提供具体的下一步建议
5. 适当使用表情符号让回复更友好

用户原始问题：{user_query}
当前系统状态：{system_status['detection_status']} (置信度{system_status['confidence']}%)
"""

def enhance_final_response(response_text: str, strategy: dict, system_status: dict) -> str:
    """增强最终回复的格式和内容"""
    
    # 添加策略相关的表情符号
    emoji_prefix = strategy['emoji']
    
    # 根据风险等级添加提示
    risk_note = ""
    if "二级预警" in system_status['detection_status']:
        risk_note = "\n\n⚡ **紧急提示**：这是高级别预警，请立即采取行动！"
    elif "一级预警" in system_status['detection_status']:
        risk_note = "\n\n🔔 **重要提醒**：请尽快安排检查，避免故障升级。"
    
    return f"{emoji_prefix} {response_text}{risk_note}"

def generate_fallback_response(user_query: str, system_status: dict, error: Exception) -> str:
    """生成降级回复"""
    
    query_lower = user_query.lower()
    
    # 智能降级回复
    if any(word in query_lower for word in ['状态', '检测', '预警']):
        status_template = RESPONSE_TEMPLATES["system_status"].format(
            compute_load="38%",
            inference_latency="15ms",
            accuracy="97.5%",
            comm_latency="45ms", 
            data_integrity="100%",
            overall_status="系统运行正常" if "正常" in system_status['detection_status'] else "系统检测到异常"
        )
        return f"🔧 **系统状态概览**\n\n{status_template}"
    
    elif any(word in query_lower for word in ['故障', '风险']):
        risk_level = "高" if "二级" in system_status['detection_status'] else "中" if "一级" in system_status['detection_status'] else "低"
        
        risk_template = RESPONSE_TEMPLATES["risk_assessment"].format(
            status=system_status['detection_status'],
            confidence=system_status['confidence'],
            risk_level=risk_level,
            analysis_basis="基于深度学习模型检测到异常电流特征",
            immediate_actions="• 检查相关回路连接点\n• 监测温度变化\n• 准备维护工具",
            next_steps="• 生成详细诊断报告\n• 创建维护工单\n• 安排预防性检查"
        )
        return risk_template
    
    else:
        return f"""🤖 **智能助手回复**

抱歉，系统暂时遇到技术问题，但我仍能为您提供帮助。

**当前系统状态**：
• 检测状态：{system_status['detection_status']}
• 置信程度：{system_status['confidence']}%
• 监测位置：{system_status['circuit_id']}

💡 **建议操作**：
1. 如遇紧急情况，立即通知轮机长
2. 对于技术问题，可尝试重新提问
3. 或联系技术支持：船岸通信频道

错误详情：{str(error)}
"""

def gemini_agent_response(user_query: str, system_status: dict):
    """增强的智能体响应函数 - 完全重写"""
    client = get_gemini_client()
    
    if client is None:
        return "⚠️ Gemini 客户端未初始化（可能缺少 API Key），无法执行 AI 推理。请检查配置。"
    
    # 1. 分析查询类型和紧急程度
    query_type = analyze_query_type(user_query, system_status)
    strategy = RESPONSE_STRATEGIES.get(query_type, RESPONSE_STRATEGIES["status_query"])
    
    # 2. 构建增强的系统上下文
    enhanced_context = build_enhanced_context(system_status, query_type)
    
    # 3. 构建专业提示词
    system_instruction = build_system_instruction(strategy, system_status)
    
    # 4. 完整的提示词
    full_prompt = f"""
{system_instruction}

{enhanced_context}

用户提问：{user_query}

请基于以上信息，用{strategy['tone']}的语气，按照{strategy['structure']}的结构进行回复。
"""
    
    try:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=list(AVAILABLE_TOOLS.values()),
        )
        
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=full_prompt,
            config=config,
        )
        
        # 处理工具调用
        final_response = None
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call'):
                            function_call = part.function_call
                            tool_name = function_call.name
                            tool_args = dict(function_call.args)
                            
                            if tool_name in AVAILABLE_TOOLS:
                                # 增强工具参数
                                enhanced_args = enhance_tool_arguments(tool_name, tool_args, system_status)
                                
                                try:
                                    tool_result = AVAILABLE_TOOLS[tool_name](**enhanced_args)
                                    
                                    # 基于工具结果生成智能回复
                                    tool_prompt = build_tool_response_prompt(
                                        user_query, tool_name, tool_result, system_status, strategy
                                    )
                                    
                                    tool_response = client.models.generate_content(
                                        model='gemini-2.0-flash-exp',
                                        contents=tool_prompt,
                                    )
                                    
                                    final_response = enhance_final_response(tool_response.text, strategy, system_status)
                                    
                                except Exception as tool_error:
                                    return f"⚠️ 工具执行失败：{tool_error}\n\n请尝试重新提问或联系技术支持。"
        
        # 如果工具调用返回了结果，使用它；否则使用原始响应
        if final_response:
            return final_response
        else:
            return enhance_final_response(response.text, strategy, system_status)
            
    except Exception as e:
        return generate_fallback_response(user_query, system_status, e)

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
        st.session_state.early_arc_confidence = 70.0 

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
        
        st.subheader("系统信息")
        st.info("""
        **架构层级:**
        - 🚢 船端边缘计算
        - ☁️ 岸基智能体
        - 🔗 船岸协同
        """)

    col1, col2 = st.columns([3, 2])

    # --- 实时监测 Dashboard ---
    with col1:
        st.header("📊 实时监测 Dashboard")
        
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
        
        # 1. 状态显示
        st.markdown(
            f"**检测状态:** <span style='color:{color}; font-size: 24px;'>{status_text}</span>",
            unsafe_allow_html=True
        )
        
        # 2. Metric
        st.metric("模型置信度", f"{confidence:.1f}%")
        st.metric("监测回路", st.session_state.circuit_id)

        # 3. 波形图
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(t_series, current_data, label=f'Current Waveform (A) @ {system_status["timestamp"]}', color=color, linewidth=1)
        ax.set_title(f" Real-time current waveform monitoring ")
        ax.set_xlabel("Time(ms)")
        ax.set_ylabel("Current(A)")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylim(-20, 20)
        ax.legend(loc='upper right')
        
        if st.session_state.fault_scenario == "early_arc":
            ax.plot(t_series, current_data + 2, label='Informer Predicted Risk Trend', color='purple', linestyle='--', alpha=0.7)
            ax.legend(loc='upper right')
        
        st.pyplot(fig)
        plt.close(fig)
        
        # 4. 预警/提示信息
        if "预警" in status_text:
            st.warning(f"🚨 **{status_text}** - 模型置信度 {confidence:.1f}%，请立即启动智能体进行诊断!")
        elif "干扰" in status_text:
            st.info("ℹ️ **干扰信号** - 检测到瞬时高频，判断为电机启动，请持续监测。")
        else:
            st.success("✅ **运行正常** - 系统稳定，故障率低。")

    # --- 智能体交互中心 ---
    with col2:
        st.header("💬 智能体交互中心")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        st.subheader("💡 快捷指令")
        presets = {
            "🚨 紧急诊断": "03号舱回路检测到异常，请立即分析风险等级和应对措施！",
            "🔧 故障分析": "请详细分析当前故障的根本原因和维修方案",
            "🛠️ 维护指导": "根据当前预警级别，生成具体的维护作业指导",
            "📊 系统状态": "请全面评估系统运行状态和健康度"
        }
        
        preset_cols = st.columns(2)
        
        for i, (preset_name, preset_text) in enumerate(presets.items()):
            col = preset_cols[i % 2]
            if col.button(f"{preset_name}", key=preset_name, use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": preset_text})
                
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
                
                # 强制 Rerun 以确保界面和状态完全同步
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
            
    # --- 定时刷新机制 ---
    time_spent = time.time() - st.session_state.last_update 
    sleep_time = max(0, 0.5 - time_spent)
    time.sleep(sleep_time)
    st.rerun()

if __name__ == "__main__":
    main()
