# face.py
import streamlit as st
import config
from utils.helpers import resource_path

# ==================== 页面全局配置 ====================
st.set_page_config(
    page_title="河海大学 · AI算法可视化学习平台",
    page_icon="🧠",
    layout="wide",
    menu_items={}
)

# 注入全局 CSS（包含原有样式 + 强制不换行）
st.markdown(config.GLOBAL_CSS, unsafe_allow_html=True)
# 追加一条专门控制侧边栏 radio 选项不换行的样式
st.markdown("""
<style>
    /* 侧边栏单选按钮文字不换行，并适当缩小字体防止溢出 */
    [data-testid="stSidebar"] label[data-baseweb="radio"] span {
        white-space: nowrap;
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== 初始化 session_state ====================
if "current_page" not in st.session_state:
    st.session_state.current_page = "home"

# ==================== 顶部标题栏 ====================
col1, col2, col3 = st.columns([1, 4, 1])
with col1:
    logo_path = resource_path("assets/hhu_logo.png")
    st.image(logo_path, width=60)
with col2:
    st.markdown(
        '<div style="display: flex; align-items: baseline;"><span class="school-name">河海大学</span><span class="platform-title">AI算法可视化学习平台</span></div>',
        unsafe_allow_html=True)
with col3:
    st.markdown(
        f'<div style="text-align: right; padding-top: 10px;"><span style="color: {config.HHU_BLUE};">💧 水之子 · 智未来</span></div>',
        unsafe_allow_html=True)
st.markdown('<div class="top-gradient"></div>', unsafe_allow_html=True)

# ==================== 左侧导航栏（带圆点指示器的 radio） ====================
with st.sidebar:
    # 校徽
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.image(resource_path("assets/hhu_logo.png"), width=160)
    st.markdown(
        f'<p style="color: {config.HHU_BLUE}; font-weight: bold; text-align: center; margin-top: 5px;">Hohai University</p>',
        unsafe_allow_html=True)
    st.markdown("---")

    # 机器学习模块
    st.markdown('<div class="sidebar-header">📁 机器学习展示</div>', unsafe_allow_html=True)
    ml_options = ["📊 分类", "📈 回归", "🔵 聚类"]
    ml_index = 0
    if st.session_state.current_page == "classification":
        ml_index = 0
    elif st.session_state.current_page == "regression":
        ml_index = 1
    elif st.session_state.current_page == "clustering":
        ml_index = 2
    else:
        ml_index = 0

    ml_selected = st.radio(
        "机器学习导航",
        ml_options,
        index=ml_index if st.session_state.current_page in ["classification", "regression", "clustering"] else None,
        key="ml_radio",
        label_visibility="collapsed"
    )
    if ml_selected:
        page_map = {"📊 分类": "classification", "📈 回归": "regression", "🔵 聚类": "clustering"}
        new_page = page_map[ml_selected]
        if st.session_state.current_page != new_page:
            st.session_state.current_page = new_page
            # st.rerun()

    # 神经网络模块（仅保留四个板块）
    st.markdown('<div class="sidebar-header" style="margin-top: 20px;">🧬 神经网络动画展示</div>', unsafe_allow_html=True)
    nn_options = [
        "🔷 基础神经网络动画演示",
        "🧩 卷积神经网络 (CNN)",
        "🔄 循环神经网络 (RNN)",
        "👁 注意力机制（Transformer）"
    ]
    nn_page_map = {
        "🔷 基础神经网络动画演示": "nn_basic",
        "🧩 卷积神经网络 (CNN)": "nn_cnn",
        "🔄 循环神经网络 (RNN)": "nn_rnn",
        "👁 注意力机制（Transformer）": "nn_attention"
    }
    current_nn_page = st.session_state.current_page
    nn_index = 0
    if current_nn_page in nn_page_map.values():
        nn_index = list(nn_page_map.values()).index(current_nn_page)
    else:
        nn_index = 0

    nn_selected = st.radio(
        "神经网络导航",
        nn_options,
        index=nn_index if current_nn_page in nn_page_map.values() else None,
        key="nn_radio",
        label_visibility="collapsed"
    )
    if nn_selected:
        new_page = nn_page_map[nn_selected]
        if st.session_state.current_page != new_page:
            st.session_state.current_page = new_page
            # st.rerun()

    # 国产大模型模块
    st.markdown('<div class="sidebar-header" style="margin-top: 20px;">🤖 国产大模型</div>', unsafe_allow_html=True)
    llm_options = ["🔥 DeepSeek", "🌟 智谱GLM", "💫 通义千问", "⭐ 文心一言"]
    llm_page_map = {
        "🔥 DeepSeek": "llm_deepseek",
        "🌟 智谱GLM": "llm_zhipu",
        "💫 通义千问": "llm_qwen",
        "⭐ 文心一言": "llm_wenxin"
    }
    current_llm_page = st.session_state.current_page
    llm_index = 0
    if current_llm_page in llm_page_map.values():
        llm_index = list(llm_page_map.values()).index(current_llm_page)
    else:
        llm_index = 0

    llm_selected = st.radio(
        "大模型导航",
        llm_options,
        index=llm_index if current_llm_page in llm_page_map.values() else None,
        key="llm_radio",
        label_visibility="collapsed"
    )
    if llm_selected:
        new_page = llm_page_map[llm_selected]
        if st.session_state.current_page != new_page:
            st.session_state.current_page = new_page
            # st.rerun()

    # 返回首页按钮（保留，方便操作）
    st.markdown("---")
    st.markdown('<a href="/" target="_self" style="text-decoration:none; color:#1A7EC1;">🏠 返回首页</a>', unsafe_allow_html=True)
        # st.rerun()

    # 校训
    st.markdown("---")
    st.markdown("""
    <div class="motto-bar">
        <div style="font-weight: bold; margin-bottom: 5px;">校训</div>
        <div>艰苦朴素 · 实事求是</div>
        <div>严格要求 · 勇于探索</div>
    </div>
    """, unsafe_allow_html=True)

    # 底部信息
    st.markdown("""
    <div style="margin-top: 30px; text-align: center; color: #999; font-size: 12px;">
        © 2026 河海大学<br>
        信息科学与工程学院
    </div>
    """, unsafe_allow_html=True)

# ==================== 右侧主内容区（路由） ====================
page = st.session_state.current_page

# ----- 机器学习模块 -----
if page == "classification":
    try:
        from pages_modules import classification_lab
        classification_lab.render_classification_lab()
    except ImportError as e:
        st.error(f"加载分类模块失败: {e}")
elif page == "regression":
    try:
        from pages_modules import regression_lab
        regression_lab.render_regression_lab()
    except ImportError as e:
        st.error(f"加载回归模块失败: {e}")
elif page == "clustering":
    try:
        from pages_modules import clustering_lab
        clustering_lab.render_clustering_lab()
    except ImportError as e:
        st.error(f"加载聚类模块失败: {e}")

# ----- 神经网络模块（四个板块） -----
elif page.startswith("nn_"):
    if page == "nn_basic":
        try:
            from pages_modules import neural_vis_module
            neural_vis_module.render_neural_network_viz()
        except ImportError as e:
            st.error(f"加载基础神经网络模块失败: {e}")
    elif page == "nn_cnn":
        try:
            from pages_modules import cnn_viz_module
            cnn_viz_module.nv_render_cnn_viz()
        except ImportError as e:
            st.error(f"加载 CNN 模块失败: {e}")
    elif page == "nn_rnn":
        try:
            from pages_modules import rnn_viz_module
            rnn_viz_module.nv_render_rnn_viz()
        except ImportError as e:
            st.error(f"加载 RNN 模块失败: {e}")
    elif page == "nn_attention":
        try:
            from pages_modules import transformer_viz_module
            transformer_viz_module.nv_render_transformer_viz()
        except ImportError as e:
            st.error(f"加载 Transformer 模块失败: {e}")

# ----- 国产大模型模块 -----
elif page.startswith("llm_"):
    model_map = {
        "llm_deepseek": "DeepSeek",
        "llm_zhipu": "智谱GLM",
        "llm_qwen": "通义千问",
        "llm_wenxin": "文心一言"
    }
    model_name = model_map.get(page, "智谱GLM")
    try:
        from pages_modules import llm
        llm.show(model_preselected=model_name)
    except ImportError as e:
        st.error(f"加载大模型模块失败: {e}")

# ----- 首页 -----
else:
    try:
        from pages_modules import home
        home.show()
    except ImportError:
        st.title("🧠 欢迎使用 AI 算法可视化学习平台")
        st.markdown("通过拖拽组件、调整参数、观察实时变化，直观理解人工智能算法。")
        st.info("👈 请从左侧菜单选择模块开始学习。")

# ==================== 底部状态栏 ====================
st.markdown("---")
bottom_col1, bottom_col2 = st.columns([3, 1])
with bottom_col1:
    page_display = {
        "classification": "📊 分类",
        "regression": "📈 回归",
        "clustering": "🔵 聚类",
        "nn_basic": "🔷 基础神经网络动画演示",
        "nn_cnn": "🧩 卷积神经网络 (CNN)",
        "nn_rnn": "🔄 循环神经网络 (RNN)",
        "nn_attention": "👁 注意力机制（Transformer）",
        "llm_deepseek": "🔥 DeepSeek",
        "llm_zhipu": "🌟 智谱GLM",
        "llm_qwen": "💫 通义千问",
        "llm_wenxin": "⭐ 文心一言"
    }.get(page, "首页")
    if page != "home":
        st.markdown(f'<div class="status-bar">📍 当前页面：{page_display}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-bar">📍 当前位于首页</div>', unsafe_allow_html=True)
with bottom_col2:
    st.markdown('<div style="text-align: right; color: #888; font-size: 13px;">基于 Streamlit + Plotly + Scikit-learn</div>', unsafe_allow_html=True)