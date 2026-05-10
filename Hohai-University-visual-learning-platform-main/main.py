import os
import sys

import streamlit as st

from classification_lab import render_classification_lab
from regression_lab import render_regression_lab
from clustering_lab import render_clustering_lab


def resource_path(relative_path: str) -> str:
    """兼容开发环境与打包环境的资源路径。"""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


st.set_page_config(
    page_title="河海大学 · AI算法可视化学习平台",
    page_icon="🧠",
    layout="wide",
    menu_items={},
)


def inject_global_css():
    st.markdown(
        """
        <style>
            html, body, [class*="css"] {
                font-family: "Microsoft YaHei", "PingFang SC", "SimHei", sans-serif;
            }
            :root {
                --hhu-blue: #1A7EC1;
                --hhu-blue-light: #3A9BDC;
                --hhu-blue-dark: #0F5B9E;
                --accent-orange: #F39C12;
                --text-dark: #22313F;
                --text-gray: #5B6B79;
                --bg-light: #F8FAFC;
                --card-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
            }
            .top-gradient {
                height: 2px;
                background: linear-gradient(90deg, var(--hhu-blue), var(--hhu-blue-light), var(--hhu-blue));
                margin-top: 8px;
                margin-bottom: 20px;
            }
            .school-name {
                font-size: 32px;
                font-weight: 700;
                color: var(--hhu-blue-dark);
                letter-spacing: 4px;
            }
            .platform-title {
                font-size: 18px;
                color: var(--text-gray);
                margin-left: 14px;
                padding-left: 14px;
                border-left: 2px solid #d8e4ef;
            }
            .sidebar-header {
                font-size: 16px;
                font-weight: 700;
                color: var(--hhu-blue-dark);
                margin: 18px 0 12px 0;
                padding-bottom: 6px;
                border-bottom: 2px solid var(--hhu-blue-light);
            }
            .welcome-box {
                background: linear-gradient(135deg, #ffffff 0%, var(--bg-light) 100%);
                border: 1px solid #E2E8F0;
                border-radius: 20px;
                padding: 28px 30px;
                text-align: center;
                box-shadow: var(--card-shadow);
            }
            .welcome-title {
                font-size: 28px;
                color: var(--hhu-blue-dark);
                margin-bottom: 10px;
                font-weight: 700;
            }
            .guide-card, .preview-card, .notice-card {
                background: #ffffff;
                border-radius: 16px;
                padding: 22px 20px;
                border: 1px solid #E8EEF5;
                box-shadow: var(--card-shadow);
                height: 100%;
            }
            .guide-card:hover, .preview-card:hover {
                box-shadow: 0 8px 20px rgba(26, 126, 193, 0.08);
                border-color: var(--hhu-blue-light);
                transform: translateY(-2px);
                transition: all 0.2s ease;
            }
            .motto-bar {
                background: linear-gradient(135deg, var(--hhu-blue-dark) 0%, var(--hhu-blue) 100%);
                color: white;
                padding: 14px 20px;
                border-radius: 12px;
                margin: 20px 0;
                text-align: center;
                font-size: 15px;
                letter-spacing: 2px;
                box-shadow: var(--card-shadow);
            }
            .status-bar {
                background: var(--bg-light);
                border-radius: 30px;
                padding: 8px 24px;
                color: var(--hhu-blue-dark);
                font-size: 14px;
                border: 1px solid #E2E8F0;
                display: inline-block;
            }
            .placeholder-box {
                background: linear-gradient(135deg, #ffffff 0%, #f7fbff 100%);
                border: 1px solid #dbe8f5;
                border-radius: 20px;
                padding: 32px 28px;
                box-shadow: var(--card-shadow);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    col1, col2, col3 = st.columns([1, 4, 1])

    with col1:
        st.markdown(
            '<div style="font-size: 38px; padding-top: 6px; text-align: center;">🌊</div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style="display: flex; align-items: baseline; flex-wrap: wrap;">
                <span class="school-name">河海大学</span>
                <span class="platform-title">AI算法可视化学习平台</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            '<div style="text-align: right; padding-top: 10px; color: #1A7EC1; font-weight: 600;">校训 · 实事求是</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="top-gradient"></div>', unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        logo_path = resource_path("hhu_logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, width=170)
        else:
            st.markdown(
                '<p style="color: #1A7EC1; font-weight: 700; text-align: center; font-size: 20px; margin-bottom: 8px;">Hohai University</p>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<p style="color: #1A7EC1; font-weight: bold; text-align: center; margin-top: 0; margin-bottom: 0;">Hohai University</p>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        st.markdown('<div class="sidebar-header">📘 机器学习</div>', unsafe_allow_html=True)
        ml_option = st.radio(
            "",
            ["📊 分类", "📈 回归", "🔵 聚类"],
            label_visibility="collapsed",
            index=None,
            key="ml_radio",
        )

        st.markdown('<div class="sidebar-header">🧩 神经网络</div>', unsafe_allow_html=True)
        nn_option = st.radio(
            "",
            [
                "🔷 前馈神经网络",
                "🟠 BP神经网络",
                "🟢 卷积神经网络（CNN）",
                "🟣 循环神经网络（RNN）",
                "👁 注意力机制",
                "🏛 深度神经网络（DNN）",
                "⚡ 生成对抗网络（GAN）",
            ],
            label_visibility="collapsed",
            index=None,
            key="nn_radio",
        )

        st.markdown('<div class="sidebar-header">🤖 国产大模型</div>', unsafe_allow_html=True)
        llm_option = st.radio(
            "",
            ["🔥 DeepSeek", "✨ 智谱GLM", "🟡 通义千问", "⭐ 文心一言"],
            label_visibility="collapsed",
            index=None,
            key="llm_radio",
        )

        st.markdown("---")
        st.markdown(
            """
            <div class="motto-bar">
                <div style="font-weight: bold; margin-bottom: 6px;">校训</div>
                <div>艰苦朴素 · 实事求是</div>
                <div>严格要求 · 勇于探索</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="margin-top: 30px; text-align: center; color: #999; font-size: 12px;">
                © 河海大学<br>
                信息科学与工程学院
            </div>
            """,
            unsafe_allow_html=True,
        )

    return ml_option, nn_option, llm_option


def render_home_page():
    st.markdown(
        """
        <div class="welcome-box">
            <div style="font-size: 36px; margin-bottom: 10px; color: #1A7EC1;">🧠</div>
            <div class="welcome-title">欢迎使用 AI 算法可视化学习平台</div>
            <p style="font-size: 16px; color: #555; max-width: 700px; margin: 15px auto 0 auto; line-height: 1.8;">
                通过调节参数、观察实时变化、结合教学解释，
                直观理解机器学习、神经网络与大模型相关算法的工作原理。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)

    st.markdown("### 快速开始")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="guide-card">
                <div style="font-size: 32px; margin-bottom: 10px;">1️⃣</div>
                <h4 style="color: #0F5B9E; margin-bottom: 8px;">选择学习模块</h4>
                <p style="color: #666; font-size: 14px;">
                    从左侧进入分类、回归、聚类等模块，先建立算法整体认识。
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="guide-card">
                <div style="font-size: 32px; margin-bottom: 10px;">2️⃣</div>
                <h4 style="color: #0F5B9E; margin-bottom: 8px;">调节参数观察变化</h4>
                <p style="color: #666; font-size: 14px;">
                    利用滑块、单选框和按钮观察模型边界、拟合曲线和聚类结构如何变化。
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="guide-card">
                <div style="font-size: 32px; margin-bottom: 10px;">3️⃣</div>
                <h4 style="color: #0F5B9E; margin-bottom: 8px;">结合图像理解原理</h4>
                <p style="color: #666; font-size: 14px;">
                    每个模块都配有中文教学说明，帮助把“图像现象”和“算法原理”对应起来。
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    st.markdown("### 模块预览")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            """
            <div class="preview-card">
                <div style="font-size: 40px; margin-bottom: 12px;">📘</div>
                <h4 style="color: #0F5B9E;">机器学习</h4>
                <p style="color: #555; font-size: 14px; line-height: 1.8;">
                    <b>分类</b>：观察决策边界与算法差异<br>
                    <b>回归</b>：观察拟合曲线与误差变化<br>
                    <b>聚类</b>：观察无监督分组与评价指标
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """
            <div class="preview-card">
                <div style="font-size: 40px; margin-bottom: 12px;">🧩</div>
                <h4 style="color: #0F5B9E;">神经网络</h4>
                <p style="color: #555; font-size: 14px; line-height: 1.8;">
                    前馈网络、BP、CNN、RNN、注意力机制等模块可继续扩展接入。
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            """
            <div class="preview-card">
                <div style="font-size: 40px; margin-bottom: 12px;">🤖</div>
                <h4 style="color: #0F5B9E;">国产大模型</h4>
                <p style="color: #555; font-size: 14px; line-height: 1.8;">
                    可继续整合 DeepSeek、智谱 GLM、通义千问、文心一言 等能力体验。
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)

    st.markdown("### 当前机器学习模块进展")
    st.success("分类、回归、聚类三个教学模块已接入，可直接从左侧“机器学习”菜单进入。")


def render_placeholder_page(title: str, section_name: str):
    st.markdown(
        f"""
        <div class="placeholder-box">
            <h2 style="color: #0F5B9E; margin-top: 0;">{title}</h2>
            <p style="font-size: 16px; color: #4f5d6b; line-height: 1.8;">
                当前进入的是 <b>{section_name}</b> 区域。
                这部分可以继续由对应负责人在后续分支中接入独立页面。
            </p>
            <p style="font-size: 15px; color: #6b7885; margin-bottom: 0;">
                机器学习模块已经完成接入；如需联调，建议先保证首页、机器学习、其他模块入口互不冲突。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def run_app():
    inject_global_css()
    render_header()
    ml_option, nn_option, llm_option = render_sidebar()

    if ml_option == "📊 分类":
        render_classification_lab()
        return

    if ml_option == "📈 回归":
        render_regression_lab()
        return

    if ml_option == "🔵 聚类":
        render_clustering_lab()
        return

    if nn_option:
        render_placeholder_page(nn_option, "神经网络")
        return

    if llm_option:
        render_placeholder_page(llm_option, "国产大模型")
        return

    render_home_page()


if __name__ == "__main__":
    run_app()
