import streamlit as st
import datetime
import os
import sys
from news_analyzer_crew import run_news_analysis, EV_KEYWORDS
import validators
import re
from dotenv import load_dotenv, find_dotenv, set_key
import markdown

# 加载环境变量
load_dotenv()

# 验证邮箱地址（支持多个）
def validate_emails(email_str):
    """
    验证一个或多个邮箱地址是否有效
    
    参数:
    - email_str: 邮箱地址字符串，可以是单个地址或用逗号、分号或空格分隔的多个地址
                 支持"邮箱:授权码"格式
    
    返回:
    - 所有邮箱有效返回True，任一邮箱无效返回False
    """
    if not email_str:
        return False
        
    # 分割邮箱地址
    emails = [e.strip() for e in re.split(r'[,;\s]+', email_str) if e.strip()]
    
    if not emails:
        return False
        
    # 验证每个邮箱地址
    email_pattern = r"[^@]+@[^@]+\.[^@]+"
    for email in emails:
        # 如果包含授权码格式(email:password)，只验证邮箱部分
        if ':' in email:
            email = email.split(':', 1)[0].strip()
            
        if not re.match(email_pattern, email):
            return False
            
    return True

# 保存设置到.env文件
def save_settings_to_env(settings_dict):
    """
    将设置保存到.env文件
    
    参数:
    - settings_dict: 包含设置键值对的字典
    
    返回:
    - 成功返回True，失败返回错误信息
    """
    try:
        dotenv_file = find_dotenv()
        if not dotenv_file:
            # 如果.env文件不存在，创建一个
            dotenv_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
            with open(dotenv_file, "w") as f:
                f.write("# 自动生成的配置文件\n")
        
        # 将每个设置保存到.env文件
        for key, value in settings_dict.items():
            if value:  # 只保存非空值
                set_key(dotenv_file, key, value)
        
        return True
    except Exception as e:
        return str(e)

def main():
    # 确保set_page_config是第一个Streamlit命令
    st.set_page_config(
        page_title="AI新能源汽车全球分析助手",
        page_icon="🚗",
        layout="wide"
    )
    
    # 设置自定义CSS样式
    st.markdown("""
    <style>
    /* 程序员主题颜色 */
    :root {
        --bg-color: #1E1E1E;
        --sidebar-color: #0D1117;
        --primary-color: #007ACC;
        --secondary-color: #6B9FFF;
        --accent-color: #569CD6;
        --success-color: #4EC9B0;
        --warning-color: #CE9178;
        --error-color: #F14C4C;
        --text-color: #E9E9E9;
        --text-secondary: #CCCCCC;
        --card-bg: #252526;
        --input-bg: #3C3C3C;
        --button-color: #0E639C;
    }
    
    /* 整体页面样式 */
    .stApp {
        background-color: var(--bg-color) !important;
        color: var(--text-color);
    }
    
    /* 隐藏Streamlit默认页脚 */
    footer {
        visibility: hidden;
    }
    
    /* 顶部标题栏 */
    .stApp header {
        background-color: var(--bg-color);
        border-bottom: 1px solid #333333;
    }
    
    /* 侧边栏样式 */
    [data-testid="stSidebar"] {
        background-color: var(--sidebar-color);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background-color: var(--sidebar-color);
    }
    
    /* 侧边栏内容颜色 */
    [data-testid="stSidebarUserContent"] {
        background-color: var(--sidebar-color);
    }
    
    /* 确保侧边栏文字为白色 */
    .stApp [data-testid="stSidebarUserContent"] p, 
    .stApp [data-testid="stSidebarUserContent"] h1, 
    .stApp [data-testid="stSidebarUserContent"] h2, 
    .stApp [data-testid="stSidebarUserContent"] h3, 
    .stApp [data-testid="stSidebarUserContent"] h4,
    .stApp [data-testid="stSidebarUserContent"] li {
        color: var(--text-color) !important;
    }
    
    /* 标题文字样式 */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-color) !important;
        font-weight: 600;
    }
    
    /* 主标题特殊样式 */
    h1 {
        color: var(--text-color) !important;
        font-weight: 700;
        border-bottom: 2px solid var(--primary-color);
        padding-bottom: 0.5rem;
        margin-bottom: 2rem;
        display: inline-block;
    }
    
    h2 {
        color: var(--text-color) !important;
    }
    
    h3 {
        color: var(--secondary-color) !important;
    }
    
    /* 段落文字样式 */
    p, span, div {
        color: var(--text-color);
    }
    
    /* 链接样式 */
    a {
        color: var(--secondary-color) !important;
        text-decoration: none;
    }
    
    a:hover {
        text-decoration: underline;
    }
    
    /* 卡片容器样式 */
    .card-container {
        background-color: var(--card-bg);
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
        border: 1px solid #383838;
    }
    
    /* 主表单样式 */
    .main-form {
        background-color: var(--card-bg);
        border-radius: 8px;
        padding: 2rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        border: 1px solid #383838;
    }
    
    /* 按钮容器样式 */
    .button-container {
        padding: 0.5rem;
    }
    
    /* 按钮基础样式 */
    .stButton > button {
        background-color: var(--button-color) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        border: none;
        width: 100%;
        transition: all 0.2s ease;
        text-shadow: 0px 1px 2px rgba(0, 0, 0, 0.15);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.15);
        outline: none !important;
    }
    
    .stButton > button:hover {
        background-color: #1177BB !important;
        transform: translateY(-1px);
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    /* 品牌按钮 */
    .brand-button .stButton > button {
        background-color: #2C58A0 !important;
    }
    
    .brand-button .stButton > button:hover {
        background-color: #3469B8 !important;
    }
    
    /* 时间按钮 */
    .time-button .stButton > button {
        background-color: #2D7A49 !important;
    }
    
    .time-button .stButton > button:hover {
        background-color: #3A9D5E !important;
    }
    
    /* 技术按钮 */
    .tech-button .stButton > button {
        background-color: #7E236D !important;
    }
    
    .tech-button .stButton > button:hover {
        background-color: #9D3489 !important;
    }
    
    /* 提交按钮 */
    .stForm [data-testid="stForm"] button[type="submit"] {
        background-color: #C04C38 !important;
        font-size: 1.1em !important;
        padding: 0.6rem 1.2rem !important;
        margin-top: 1.5rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em;
    }
    
    .stForm [data-testid="stForm"] button[type="submit"]:hover {
        background-color: #D15A44 !important;
    }
    
    /* 输入框样式优化 */
    [data-testid="stTextInput"] > div > div {
        background-color: var(--input-bg) !important;
        border: 1px solid #555555 !important;
        border-radius: 6px !important;
        color: white !important;
    }
    
    [data-testid="stTextInput"] > label {
        color: var(--text-color) !important;
    }
    
    /* 复选框样式 */
    [data-testid="stCheckbox"] {
        color: var(--text-color) !important;
    }
    
    [data-testid="stCheckbox"] > label > div[role="checkbox"] {
        border-color: #555555 !important;
    }
    
    /* 下拉选择框样式 */
    [data-testid="stSelectbox"] > div > div {
        background-color: var(--input-bg) !important;
        border: 1px solid #555555 !important;
        border-radius: 6px !important;
        color: white !important;
    }
    
    /* 日期选择器 */
    .stDateInput > div > div > input {
        background-color: var(--input-bg) !important;
        border: 1px solid #555555 !important;
        border-radius: 6px !important;
        color: white !important;
    }
    
    /* 滑动条 */
    [data-testid="stSlider"] > div > div {
        background-color: var(--primary-color) !important;
    }
    
    [data-testid="stSlider"] > div > div > div > div {
        background-color: var(--secondary-color) !important;
    }
    
    /* 提示信息优化 */
    .stAlert {
        background-color: #2A3749 !important;
        color: white !important;
        border: 1px solid #3C506B !important;
    }
    
    .stAlert > div {
        color: white !important;
    }
    
    /* 成功消息 */
    [data-baseweb="notification"] {
        background-color: #1E3C2F !important;
        border-color: #2D5D47 !important;
    }
    
    /* 错误消息 */
    [data-baseweb="notification"][kind="negative"] {
        background-color: #3F1E1E !important;
        border-color: #5D2D2D !important;
    }
    
    /* 折叠面板样式 */
    [data-testid="stExpander"] {
        background-color: var(--card-bg) !important;
        border: 1px solid #383838 !important;
    }
    
    [data-testid="stExpander"] summary {
        color: var(--text-color) !important;
    }
    
    /* 进度条/加载动画 */
    [role="progressbar"] > div {
        border-color: var(--primary-color) !important;
    }
    
    /* 隐藏主界面上方的搜索框/工具条 */
    [data-testid="stToolbar"] {
        visibility: hidden !important;
    }
    
    /* 隐藏界面上方可能出现的Streamlit默认元素 */
    .stDecoration {
        display: none !important;
    }
    
    /* 隐藏可能存在的空白框区域 */
    .element-container:has(+ .element-container:has(div[data-testid="stMarkdownContainer"] h2:first-of-type)) {
        display: none !important;
    }
    
    /* 隐藏主界面上方的空白框 */
    [data-testid="InputInstructions"] {
        display: none !important;
    }
    
    /* 更多可能需要隐藏的元素 */
    .stSearchButton, .stSearchOptionButton, .stSearchButtonUnselected {
        display: none !important;
    }
    
    /* 主界面顶部区域的优化 */
    .main .block-container {
        padding-top: 2rem !important;
        margin-top: 0 !important;
    }
    
    /* 针对性移除标题下方的搜索框 */
    .st-emotion-cache-16txtl3 {
        display: none !important;
    }
    
    /* 移除可能的标题下方元素 */
    .st-emotion-cache-16idsys {
        display: none !important;
    }
    
    /* 修复标题和内容之间的间距 */
    h1 + div {
        margin-top: -1.5rem !important;
    }
    
    /* 隐藏特定的搜索框相关元素 */
    div[data-baseweb="input"], 
    div[data-baseweb="base-input"],
    [data-testid="stSearch"] {
        display: none !important;
    }
    
    /* 修复布局，确保标题紧接着内容 */
    .stApp [data-testid="stAppViewContainer"] > div:first-child > div:first-child + div {
        margin-top: 0 !important;
    }
    
    /* 更加突出的输入框样式 */
    .prominent-input [data-testid="stTextInput"] > div > div {
        background-color: #3C3C3C !important;
        border: 2px solid #f44336 !important;
        border-radius: 8px !important;
        padding: 8px !important;
        margin-bottom: 10px !important;
    }
    
    /* 增加输入框高度 */
    .prominent-input input {
        height: 45px !important;
        font-size: 16px !important;
    }
    
    /* 确保输入框标签正确显示 */
    [data-testid="stTextInput"] > label,
    [data-testid="stSelectbox"] > label,
    [data-testid="stDateInput"] > label,
    [data-testid="stCheckbox"] > label {
        display: block !important;
        visibility: visible !important;
        height: auto !important;
        color: var(--text-color) !important;
        margin-bottom: 5px !important;
        font-weight: 500 !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        position: relative !important;
        z-index: 1 !important;
    }
    
    /* 确保输入框始终可见 */
    [data-testid="stTextInput"] > div > div,
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stDateInput"] > div > div {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        background-color: var(--input-bg) !important;
        border: 1px solid #555555 !important;
        border-radius: 6px !important;
        color: white !important;
        height: auto !important;
        min-height: 38px !important;
        position: relative !important;
        z-index: 1 !important;
    }
    
    /* 增强关键词输入框的显示效果 */
    .prominent-input [data-testid="stTextInput"] > div > div {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
        background-color: #3C3C3C !important;
        border: 2px solid #f44336 !important;
        border-radius: 8px !important;
        padding: 8px !important;
        margin-bottom: 10px !important;
    }

    /* 确保标签文本可见 */
    .stTextInput > label,
    .stSelectbox > label,
    .stDateInput > label,
    .stCheckbox > label {
        color: white !important;
        opacity: 1 !important;
        visibility: visible !important;
        height: auto !important;
        display: block !important;
        position: relative !important;
        z-index: 1 !important;
    }

    /* 强制显示所有输入元素及其容器 */
    .stTextInput, .stSelectbox, .stDateInput, .stCheckbox,
    div[data-baseweb="input"], div[data-baseweb="base-input"] {
        opacity: 1 !important;
        visibility: visible !important;
        display: block !important;
        margin-bottom: 10px !important;
        position: relative !important;
        z-index: 1 !important;
    }
    
    /* 输入框内文本颜色 */
    input, select, textarea {
        color: white !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    /* 确保表单内的所有输入有足够的间距 */
    form [data-testid="stForm"] [data-testid="stVerticalBlock"] > div {
        margin-bottom: 15px !important;
    }

    /* 强制保持标签位置可见 */
    .css-16huue1, .css-18e3th9 {
        position: static !important;
        visibility: visible !important;
        height: auto !important;
    }
    
    /* 修复隐藏搜索框相关元素的代码，确保不影响输入框 */
    div[data-baseweb="input"], 
    div[data-baseweb="base-input"] {
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    /* 只隐藏特定的搜索框，不影响输入框 */
    [data-testid="stSearch"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 使用自定义HTML直接创建标题区域，避免Streamlit默认组件可能产生的空白框
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h1 style="color: #E9E9E9; font-weight: 700; border-bottom: 2px solid #007ACC; padding-bottom: 0.5rem; display: inline-block;">🚗 AI新能源汽车全球分析助手</h1>
        <p style="color: #E9E9E9; margin-top: 1rem;">这个应用可以收集和分析全球范围内与新能源汽车相关的内容，深度理解行业动态、消费者洞察和技术趋势等信息。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 侧边栏设置
    with st.sidebar:
        st.header("🚗 AI新能源汽车全球分析助手")
        st.markdown("本工具可以帮助您快速了解新能源汽车领域的最新动态：")
        st.markdown("- 全球行业趋势和市场动向")
        st.markdown("- 主要汽车厂商战略和发展")  
        st.markdown("- 技术创新与突破")
        st.markdown("- 政策法规变化")
        st.markdown("- 消费者洞察和购买决策")
        # 添加DeepSeek标志和信息
        st.markdown("---")
        st.markdown("### 技术支持")
        st.markdown("基于[DeepSeek](https://deepseek.com)和[CrewAI](https://www.crewai.com/)技术构建")
        
        # 添加新能源汽车相关信息
        st.markdown("---")
        st.markdown("### 分析内容")
        st.markdown("""
        - 全球行业发展趋势
        - 中外主要厂商动向
        - 消费者购买决策变化
        - 国际政策分析
        - 全球技术发展情况
        - 中国与全球市场对比
        - 长安汽车专题分析
        - 新兴品牌分析
        - 前沿技术进展
        - 新能源消费者分析
        """)
        
        # 添加版权信息
        st.markdown("---")
        st.markdown("CopyRight@2025 变革与效率部（企业数据中心）", help="版权所有")

    # 初始化会话状态
    if 'keywords' not in st.session_state:
        st.session_state.keywords = ""
    if 'time_range_days' not in st.session_state:
        st.session_state.time_range_days = 30
    if 'email' not in st.session_state:
        st.session_state.email = ""
    if 'email_password' not in st.session_state:
        st.session_state.email_password = ""

    # 关键词输入框区域
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.subheader("🔍 输入分析关键词")
    st.markdown("请输入您要分析的新能源汽车相关关键词")
    
    # 关键词输入框
    st.markdown('<div class="prominent-input" style="margin-top: 15px; margin-bottom: 15px;">', unsafe_allow_html=True)
    keywords = st.text_input(
        "分析关键词", 
        value=st.session_state.keywords, 
        placeholder='例如："比亚迪"、"长安汽车"、"人形机器人"', 
        help="请输入您关注的新能源汽车相关关键词，多个关键词用空格分隔", 
        key="keyword_input", 
        label_visibility="visible"
    )
    st.session_state.keywords = keywords
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # 关闭卡片容器

    # 使用卡片容器包装时间范围区域 - 改为下拉框选择
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.subheader("⏱️ 时间范围")
    st.markdown("选择要分析的内容时间范围")
    
    # 时间范围下拉框选择
    st.markdown('<div class="prominent-input">', unsafe_allow_html=True)
    time_options = {
        "最近一周": 7,
        "最近一个月": 30,
        "最近三个月": 90,
        "最近一年": 365
    }
    selected_time_option = st.selectbox(
        "选择时间范围", 
        options=list(time_options.keys()),
        index=1,  # 默认选择"最近一个月"
        label_visibility="visible"
    )
    # 根据选择设置对应的天数
    st.session_state.time_range_days = time_options[selected_time_option]
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # 关闭卡片容器

    # 主界面表单
    st.markdown('<div class="main-form">', unsafe_allow_html=True)
    with st.form("ev_analysis_form"):
        st.subheader("📝 配置您的新能源汽车全球分析")
        
        # 这里不再需要时间范围滑块，因为已经使用下拉框选择了
        time_range_days = st.session_state.time_range_days
        
        # 邮箱设置
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.subheader("📧 设置报告接收邮箱")
        
        # 邮箱地址 - 确保输入框可见
        st.markdown('<div class="prominent-input" style="margin-top: 15px; margin-bottom: 15px;">', unsafe_allow_html=True)
        email = st.text_input(
            "邮箱地址", 
            value=st.session_state.email, 
            placeholder="example@126.com 或多个邮箱用逗号分隔", 
            help="我们将把分析报告发送到这个邮箱，支持多个邮箱地址（用逗号、分号或空格分隔，如: email1@126.com, email2@126.com）。如需为每个邮箱使用不同授权码，请使用格式：email1@126.com:授权码1, email2@126.com:授权码2", 
            key="email_input",
            label_visibility="visible"
        )
        st.session_state.email = email
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 邮箱授权码 - 确保输入框可见
        st.markdown('<div class="prominent-input" style="margin-top: 15px; margin-bottom: 15px;">', unsafe_allow_html=True)
        email_password = st.text_input(
            "邮箱授权码", 
            value=st.session_state.email_password, 
            placeholder="请输入邮箱的授权码（不是邮箱密码）", 
            help="邮箱的授权码，如果是Gmail，可以使用应用专用密码。对于多个邮箱，可以输入对应数量的授权码，用逗号、分号或空格分隔，系统将按顺序与邮箱地址匹配", 
            key="email_password_input", 
            type="password",
            label_visibility="visible"
        )
        st.session_state.email_password = email_password
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 添加SMTP服务器设置（高级选项）
        with st.expander("📌 邮箱服务器设置（高级选项）"):
            st.markdown("#### 📧 SMTP服务器配置")
            st.markdown('<div class="prominent-input" style="margin-top: 15px; margin-bottom: 15px;">', unsafe_allow_html=True)
            smtp_col1, smtp_col2 = st.columns(2)
            with smtp_col1:
                saved_host = os.getenv("EMAIL_HOST", "")
                smtp_host = st.text_input("SMTP服务器", value=saved_host, 
                                         help="默认根据邮箱地址自动选择，常见服务器：smtp.126.com(126邮箱), smtp.gmail.com(Gmail)", 
                                         label_visibility="visible", key="smtp_host_input")
            with smtp_col2:
                saved_port = os.getenv("EMAIL_PORT", "")
                try:
                    saved_port = int(saved_port) if saved_port else 465
                except:
                    saved_port = 465
                # 修改为直接输入框而非数字输入框带+-按钮
                smtp_port = st.text_input("SMTP端口", value=str(saved_port), 
                                        help="126邮箱使用465(SSL)或25(非SSL), Gmail使用587", 
                                        label_visibility="visible", key="smtp_port_input")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 自动填充网易邮箱服务器信息
            if email and "@126.com" in email and not smtp_host:
                smtp_host = "smtp.126.com"
                smtp_port = "465"
            elif email and "@163.com" in email and not smtp_host:
                smtp_host = "smtp.163.com"
                smtp_port = "465"
            elif email and "@qq.com" in email and not smtp_host:
                smtp_host = "smtp.qq.com"
                smtp_port = "465"
            
            # 添加保存设置选项
            save_settings = st.checkbox("保存邮箱设置到配置文件", value=False, 
                                       help="选中后，您的邮箱设置将保存到.env文件中，下次运行程序时将自动使用这些设置（授权码将安全保存）",
                                       label_visibility="visible")
            
            if save_settings:
                st.warning("⚠️ 注意：这将保存您的邮箱设置到本地配置文件，确保您使用的是个人电脑且环境安全。")
        
        # 发送时间选择
        st.markdown("#### 🕒 报告发送时间")
        delivery_options = ["立即发送", "每天 17:30", "每天早上 8:30"]
        delivery_time_option = st.selectbox("选择发送时间", options=delivery_options, index=0, label_visibility="visible")
        
        # 语言偏好设置
        st.markdown("#### 🌐 语言偏好")
        st.markdown("分析报告将以中文形式呈现，如原始内容包含其他语言，将自动翻译为中文")
        
        # 提交按钮
        col1, col2 = st.columns([3, 1])
        with col1:
            submit_button = st.form_submit_button("✨ 开始分析", use_container_width=True)
        with col2:
            save_settings_button = st.form_submit_button("💾 保存设置", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)  # 关闭表单卡片容器

    # 处理表单提交
    if submit_button:
        # 获取关键词
        keywords = st.session_state.keywords
        
        # 验证邮箱
        if not keywords:
            st.error("请输入分析关键词")
        elif not email:
            st.error("请输入邮箱地址以接收分析报告")
        elif not validate_emails(email):
            st.error("请输入有效的邮箱地址")
        # 验证邮箱密码
        elif not email_password:
            st.error("请输入邮箱授权码")
        else:
            # 设置处理是否继续的标志
            proceed = True
            
            # 处理多个邮箱和授权码的匹配
            if email and email_password:
                # 解析多个邮箱地址
                email_list = [e.strip() for e in re.split(r'[,;\s]+', email) if e.strip()]
                
                # 解析多个授权码（如果有多个）
                auth_list = [a.strip() for a in re.split(r'[,;\s]+', email_password) if a.strip()]
                
                # 确保邮箱地址和授权码数量匹配
                if len(email_list) > 1 and len(auth_list) > 1:
                    if len(email_list) == len(auth_list):
                        # 构建"邮箱:授权码"格式
                        email_with_auth = [f"{email_list[i]}:{auth_list[i]}" for i in range(len(email_list))]
                        email = ','.join(email_with_auth)
                        # 清空授权码环境变量，因为已经包含在email中
                        email_password = ""
                    else:
                        st.error(f"邮箱数量({len(email_list)})与授权码数量({len(auth_list)})不匹配，请确保两者数量相同")
                        proceed = False
            
            if proceed:
                # 转换发送时间选项
                if delivery_time_option == "立即发送":
                    send_time = "now"
                elif delivery_time_option == "每天 17:30":
                    now = datetime.datetime.now()
                    send_time = datetime.datetime.combine(now.date(), datetime.time(17, 30))
                    # 标记为每日定时任务
                    os.environ["SCHEDULE_DAILY"] = "true"
                    os.environ["SCHEDULE_TIME"] = "17:30"
                elif delivery_time_option == "每天早上 8:30":
                    now = datetime.datetime.now()
                    send_time = datetime.datetime.combine(now.date(), datetime.time(8, 30))
                    # 如果当前时间已经过了8:30，设置为明天
                    if now.time() > datetime.time(8, 30):
                        send_time = send_time + datetime.timedelta(days=1)
                    # 标记为每日定时任务
                    os.environ["SCHEDULE_DAILY"] = "true"
                    os.environ["SCHEDULE_TIME"] = "08:30"
                
                # 处理SMTP端口转换为数字
                try:
                    smtp_port_int = int(smtp_port) if smtp_port else 465
                except ValueError:
                    smtp_port_int = 465
                    
                # 保存邮箱密码到环境变量，供邮件发送功能使用
                os.environ["EMAIL_PASSWORD"] = email_password if email_password else ""
                
                # 如果用户提供了自定义SMTP设置，则更新环境变量
                if 'smtp_host' in locals() and smtp_host:
                    os.environ["EMAIL_HOST"] = smtp_host
                if 'smtp_port' in locals() and smtp_port:
                    os.environ["EMAIL_PORT"] = str(smtp_port_int)
                
                # 如果是网易126邮箱且未设置自定义SMTP服务器，自动配置
                if "@126.com" in email and not 'smtp_host' in locals():
                    os.environ["EMAIL_HOST"] = "smtp.126.com"
                    os.environ["EMAIL_PORT"] = "465"
                
                # 如果用户选择保存设置，则将其写入.env文件
                if save_settings:
                    settings_to_save = {
                        "EMAIL_USER": email.split(':')[0] if ':' in email else email,  # 如果包含授权码格式，只保存邮箱部分
                        "EMAIL_PASSWORD": email_password if email_password else "",
                    }
                    
                    # 如果提供了自定义SMTP设置，也保存它们
                    if 'smtp_host' in locals() and smtp_host:
                        settings_to_save["EMAIL_HOST"] = smtp_host
                    if 'smtp_port' in locals() and smtp_port:
                        settings_to_save["EMAIL_PORT"] = str(smtp_port_int)
                    
                    # 保存设置
                    save_result = save_settings_to_env(settings_to_save)
                    if save_result is True:
                        st.success("✅ 邮箱设置已成功保存！下次启动程序时将自动使用这些设置。")
                    else:
                        st.error(f"❌ 保存设置失败: {save_result}")
                
                # 更新会话状态保存用户选择的关键词和时间范围
                st.session_state.keywords = keywords
                st.session_state.time_range_days = time_range_days
                
                # 启动分析
                with st.spinner("AI正在收集和分析全球新能源汽车内容，这可能需要一段时间，请耐心等待..."):
                    try:
                        result = run_news_analysis(
                            keywords=keywords,
                            email=email,
                            time_range_days=time_range_days,
                            send_time=send_time,
                            ensure_chinese=True  # 确保报告以中文呈现
                        )
                        st.success("新能源汽车全球分析已完成！")
                        # 格式化邮箱显示，如果是多个邮箱，显示为列表
                        email_display = email.replace(',', '<br>').replace(';', '<br>').replace(' ', '<br>')
                        if '<br>' in email_display:
                            email_display = email_display.replace('<br>', '、')
                        st.markdown(f"分析报告将在 **{send_time if isinstance(send_time, datetime.datetime) else '立即'}** 发送到 **{email_display}**")
                        
                        # 在报告预览区域显示Markdown报告
                        if 'report_preview' in result:
                            st.subheader("📊 报告预览")
                            st.info("以下展示报告内容。完整报告包含专业引用、详细数据分析和权威来源URL链接。")
                            
                            # 添加切换按钮
                            view_mode = st.radio("查看模式", ["渲染视图", "Markdown源码"], horizontal=True)
                            
                            if view_mode == "Markdown源码":
                                # 显示原始Markdown代码
                                st.code(result['report_preview'], language="markdown")
                            else:
                                # 添加自定义CSS样式提升Markdown渲染效果
                                st.markdown("""
                                <style>
                                .rendered-markdown h1 {
                                    color: #1a5276;
                                    border-bottom: 2px solid #3498db;
                                    padding-bottom: 10px;
                                }
                                .rendered-markdown h2 {
                                    color: #2874a6;
                                    border-bottom: 1px solid #3498db;
                                    padding-bottom: 5px;
                                }
                                .rendered-markdown h3 {
                                    color: #2e86c1;
                                }
                                .rendered-markdown a {
                                    color: #3498db;
                                    text-decoration: none;
                                }
                                .rendered-markdown a:hover {
                                    text-decoration: underline;
                                }
                                .rendered-markdown blockquote {
                                    border-left: 4px solid #3498db;
                                    padding: 10px 20px;
                                    margin: 20px 0;
                                    background-color: #f5f9fa;
                                }
                                .rendered-markdown code {
                                    background-color: #f5f5f5;
                                    padding: 2px 4px;
                                    border-radius: 4px;
                                }
                                .rendered-markdown pre {
                                    background-color: #f5f5f5;
                                    padding: 15px;
                                    border-radius: 5px;
                                    overflow: auto;
                                }
                                .rendered-markdown table {
                                    border-collapse: collapse;
                                    width: 100%;
                                    margin: 20px 0;
                                }
                                .rendered-markdown th, .rendered-markdown td {
                                    border: 1px solid #ddd;
                                    padding: 8px 12px;
                                    text-align: left;
                                }
                                .rendered-markdown th {
                                    background-color: #f2f2f2;
                                }
                                .rendered-markdown tr:nth-child(even) {
                                    background-color: #f9f9f9;
                                }
                                </style>
                                """, unsafe_allow_html=True)
                                
                                # 渲染Markdown，将Markdown转换为HTML后显示
                                html_content = markdown.markdown(result['report_preview'], extensions=['tables', 'fenced_code'])
                                st.markdown(f'<div class="rendered-markdown">{html_content}</div>', unsafe_allow_html=True)
                            
                            # 下载链接
                            if 'report_filename' in result:
                                with open(result['report_filename'], "r", encoding="utf-8") as f:
                                    report_data = f.read()
                                
                                st.download_button(
                                    label="📥 下载完整分析报告 (Markdown格式)",
                                    data=report_data,
                                    file_name=result['report_filename'],
                                    mime="text/markdown",
                                    help="下载包含完整引用链接的专业分析报告，适合学术研究和商业决策参考"
                                )
                
                    except Exception as e:
                        st.error(f"分析过程中发生错误: {str(e)}")
                        
                    # 清除环境变量中的密码
                    if "EMAIL_PASSWORD" in os.environ:
                        del os.environ["EMAIL_PASSWORD"]
                    
                    # 清除自定义SMTP设置
                    if "EMAIL_HOST" in os.environ and smtp_host:
                        del os.environ["EMAIL_HOST"]
                    if "EMAIL_PORT" in os.environ and smtp_port:
                        del os.environ["EMAIL_PORT"]

if __name__ == "__main__":
    main() 