import streamlit as st
from predict_fun import predict_fun
import requests

# ==================== 页面基础配置 ====================
st.set_page_config(
    page_title="投满分 · 新闻多分类",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==================== 全局 CSS 注入 ====================
st.markdown("""
<style>
    /* ----- 全局字体与基础 ----- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    }

    /* ----- 隐藏 Streamlit 默认元素 ----- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {background: transparent;}

    /* ----- 主容器卡片化 ----- */
    .block-container {
        padding: 2rem 3rem;
        max-width: 960px;
    }

    /* ----- 标题区域 ----- */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.3rem;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #64748B;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* ----- 示例标签按钮 ----- */
    .example-chip {
        display: inline-block;
        padding: 0.4rem 1rem;
        margin: 0.25rem 0.4rem;
        background: #EEF2FF;
        color: #1E88E5;
        border-radius: 20px;
        font-size: 0.88rem;
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid transparent;
    }
    .example-chip:hover {
        background: #1E88E5;
        color: #FFFFFF;
        border-color: #1E88E5;
    }

    /* ----- 文本输入框美化 ----- */
    textarea[data-testid="stTextArea"] textarea,
    input[data-testid="stTextInput"] {
        border: 2px solid #E2E8F0 !important;
        border-radius: 12px !important;
        padding: 14px 18px !important;
        font-size: 1rem !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
        background: #FAFBFC !important;
    }
    textarea[data-testid="stTextArea"] textarea:focus,
    input[data-testid="stTextInput"]:focus {
        border-color: #1E88E5 !important;
        box-shadow: 0 0 0 3px rgba(30,136,229,0.12) !important;
        background: #FFFFFF !important;
        outline: none !important;
    }

    /* ----- 预测按钮 ----- */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.05rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.25s ease;
        box-shadow: 0 4px 14px rgba(30,136,229,0.35);
    }
    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(30,136,229,0.45);
    }
    div.stButton > button:active {
        transform: translateY(0);
    }

    /* ----- 结果卡片 ----- */
    .result-card {
        background: #FFFFFF;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-top: 1.5rem;
        border-left: 4px solid #1E88E5;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        animation: fadeInUp 0.4s ease;
    }
    .result-card .label {
        font-size: 0.82rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.4rem;
    }
    .result-card .class-name {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .result-card .input-preview {
        font-size: 0.9rem;
        color: #64748B;
        margin-top: 0.8rem;
        padding-top: 0.8rem;
        border-top: 1px solid #F1F5F9;
    }

    /* ----- 侧边栏样式 ----- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F8FAFC 0%, #EEF2FF 100%);
    }
    [data-testid="stSidebar"] .sidebar-content {
        padding: 1.5rem 1.2rem;
    }

    /* ----- 历史记录条目 ----- */
    .history-item {
        background: #FFFFFF;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border: 1px solid #E2E8F0;
        font-size: 0.85rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .history-item .h-text {
        color: #334155;
        font-weight: 500;
    }
    .history-item .h-class {
        color: #1E88E5;
        font-weight: 600;
        margin-top: 0.25rem;
    }

    /* ----- 入场动画 ----- */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ----- 响应式适配 ----- */
    @media (max-width: 768px) {
        .block-container { padding: 1rem 1rem; }
        .hero-title { font-size: 1.8rem; }
    }
</style>
""", unsafe_allow_html=True)

# ==================== 会话状态初始化 ====================
if "history" not in st.session_state:
    st.session_state.history = []  # 每条记录: {"text": str, "class": str}
if "text_input" not in st.session_state:
    st.session_state.text_input = ""  # 输入框内容持久化

# ==================== 侧边栏 ====================
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/news.png",
        width=64,
    )
    st.markdown("### 📰 投满分分类")
    st.caption("基于Bert的新闻多分类引擎")

    st.divider()

    # ----- 统计信息 -----
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("今日预测", len(st.session_state.history))
    with col_b:
        categories = set(h["class"] for h in st.session_state.history)
        st.metric("涉及类别", len(categories))

    st.divider()

    # ----- 历史记录 -----
    st.markdown("#### 🕓 预测历史")
    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history[-20:])):
            st.markdown(f"""
            <div class="history-item">
                <div class="h-text">📝 {h['text'][:50]}{'...' if len(h['text']) > 50 else ''}</div>
                <div class="h-class">🏷 {h['class']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.caption("暂无预测记录")

    st.divider()

    # ----- 清空历史 -----
    if st.button("🗑 清空历史记录", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ==================== 主内容区 ====================
# ----- 标题 -----
st.markdown('<div class="hero-title">📰 投满分分类项目</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">输入一段新闻内容，AI 自动识别其所属类别</div>',
    unsafe_allow_html=True,
)


# ----- 输入区域 -----
st.markdown("---")
text = st.text_area(
    "请输入新闻内容：",
    key="text_input",
    height=120,
    placeholder="在此粘贴或输入新闻文本……",
    label_visibility="collapsed",
)

# ----- 预测按钮 -----
predict_clicked = st.button("🔍 开始智能分类", use_container_width=True)

# ----- 执行预测 -----
if predict_clicked:
    if not text.strip():
        st.toast("⚠️ 请输入新闻内容后再预测", icon="⚠️")
    else:
        with st.spinner("🤖 AI 正在分析文本特征，请稍候……"):
            try:
                url = 'http://127.0.0.1:8003/predict'
                # 构造请求数据
                data = {"text": text.strip()}
                response = requests.post(url, json=data)
                result = response.json()
                pred_class = result["pred_class"]

                # 保存到历史记录（去重）
                if not st.session_state.history or st.session_state.history[-1]["text"] != text.strip():
                    st.session_state.history.append({
                        "text": text.strip(),
                        "class": pred_class,
                    })

                # 成功提示
                st.toast("✅ 预测完成！", icon="✅")

                # ----- 结果展示卡片 -----
                st.markdown(f"""
                <div class="result-card">
                    <div class="label">预测类别</div>
                    <div class="class-name">🏷 {pred_class}</div>
                    <div class="input-preview">
                        <strong>原文摘录：</strong>{text.strip()[:80]}{'...' if len(text.strip()) > 80 else ''}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.toast(f"❌ 预测失败：{e}", icon="❌")
                st.error(f"**预测失败**：{e}")

# ----- 页脚 -----
st.markdown("---")

