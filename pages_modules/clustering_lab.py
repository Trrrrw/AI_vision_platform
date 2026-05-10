import numpy as np
import streamlit as st

from pages_modules.clustering_data import (
    generate_clustering_dataset,
    get_dataset_label,
    get_dataset_options,
    get_dataset_summary,
    get_default_dataset,
)
from pages_modules.clustering_metrics import build_clustering_metrics
from pages_modules.clustering_models import ALGORITHM_LABELS, build_clusterer
from pages_modules.clustering_text import (
    algorithm_overview,
    bottom_conclusion,
    clustering_basics_sections,
    dataset_overview,
    live_summary,
    parameter_explanation,
)
from pages_modules.clustering_viz import render_analysis_visual, render_main_visual


STATE_DEFAULTS = {
    "cluster_lab_algorithm": "kmeans",
    "cluster_lab_dataset": "kmeans_blobs",
    "cluster_lab_seed": 17,
    "cluster_lab_view_nonce": 0,
    "cluster_lab_prev_algorithm": "kmeans",
}


def render_clustering_lab():
    ensure_state()
    sync_dataset_with_algorithm()
    inject_page_css()

    algorithm_key = st.session_state["cluster_lab_algorithm"]
    dataset_key = st.session_state["cluster_lab_dataset"]
    algo_info = algorithm_overview(algorithm_key)
    dataset_info = dataset_overview(dataset_key)

    st.markdown(
        '<a href="/" target="_self" style="text-decoration:none; font-size:14px; color:#1A7EC1;">🏠 返回首页</a>',
        unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        """
        <div class="lab-hero">
            <div class="lab-overline">机器学习 · 聚类</div>
            <div class="lab-title">无监督聚类可视化学习实验室</div>
            <div class="lab-subtitle">
                先选算法，再切换最适合它的教学数据，随后观察簇结构、参数变化和内部评价指标如何一起改变。
            </div>
            <div class="lab-badges">
                <span class="lab-badge">当前算法：{0}</span>
                <span class="lab-badge">当前数据：{1}</span>
            </div>
            <div class="lab-summary-grid">
                <div class="lab-summary-card"><b>算法一句话：</b>{2}</div>
                <div class="lab-summary-card"><b>数据一句话：</b>{3}</div>
            </div>
        </div>
        """.format(algo_info["title"], dataset_info["title"], algo_info["headline"], dataset_info["summary"]),
        unsafe_allow_html=True,
    )

    with st.expander("聚类基础知识", expanded=False):
        for index, (title, body) in enumerate(clustering_basics_sections()):
            st.markdown("**{0}. {1}**".format(index + 1, title))
            st.markdown(body)

    control_col, content_col = st.columns([0.95, 2.35], gap="large")
    with control_col:
        settings = render_control_panel(algorithm_key)

    lab_result = build_lab_result(
        algorithm_key=settings["algorithm_key"],
        dataset_key=settings["dataset_key"],
        sample_count=settings["sample_count"],
        noise=settings["noise"],
        params=settings["params"],
    )

    with content_col:
        render_display_panel(settings["algorithm_key"], settings["params"], lab_result)
        st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)
        render_bottom_panel(settings["algorithm_key"], settings["dataset_key"], settings["params"], lab_result)


def ensure_state():
    for key, value in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sync_dataset_with_algorithm():
    current_algorithm = st.session_state["cluster_lab_algorithm"]
    previous_algorithm = st.session_state["cluster_lab_prev_algorithm"]
    valid_options = get_dataset_options(current_algorithm)
    if previous_algorithm != current_algorithm or st.session_state["cluster_lab_dataset"] not in valid_options:
        st.session_state["cluster_lab_dataset"] = get_default_dataset(current_algorithm)
        st.session_state["cluster_lab_prev_algorithm"] = current_algorithm
        st.session_state["cluster_lab_view_nonce"] += 1


def inject_page_css():
    st.markdown(
        """
        <style>
            html, body, [class*="css"] {
                font-family: "Microsoft YaHei", "PingFang SC", "SimHei", sans-serif;
            }
            .lab-hero {
                background: linear-gradient(135deg, #ffffff 0%, #eef7ff 100%);
                border: 1px solid #dae8f6;
                border-radius: 24px;
                padding: 28px 30px;
                margin-bottom: 18px;
                box-shadow: 0 10px 24px rgba(15, 91, 158, 0.06);
            }
            .lab-overline {
                color: #1A7EC1;
                font-weight: 700;
                font-size: 14px;
                letter-spacing: 1px;
                margin-bottom: 8px;
            }
            .lab-title {
                color: #143d66;
                font-size: 32px;
                font-weight: 800;
                margin-bottom: 10px;
            }
            .lab-subtitle {
                color: #546879;
                font-size: 15px;
                line-height: 1.8;
            }
            .lab-badges {
                margin-top: 14px;
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }
            .lab-badge {
                background: #e8f3fd;
                color: #0F5B9E;
                padding: 7px 14px;
                border-radius: 999px;
                font-size: 14px;
                font-weight: 700;
            }
            .lab-summary-grid {
                margin-top: 16px;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 14px;
            }
            .lab-summary-card {
                background: rgba(255,255,255,0.88);
                border: 1px solid #e1edf7;
                border-radius: 18px;
                padding: 14px 16px;
                color: #4f6475;
                line-height: 1.8;
            }
            .metric-card {
                background: #ffffff;
                border: 1px solid #e4eef7;
                border-radius: 18px;
                padding: 16px 18px;
                box-shadow: 0 8px 18px rgba(15, 91, 158, 0.05);
                min-height: 102px;
            }
            .metric-label {
                color: #5c7082;
                font-size: 13px;
                margin-bottom: 8px;
            }
            .metric-value {
                color: #143d66;
                font-size: 28px;
                font-weight: 800;
            }
            .teach-note {
                background: #f8fbff;
                border: 1px solid #dbe9f7;
                border-radius: 16px;
                padding: 14px 16px;
                color: #53697b;
                line-height: 1.8;
                margin-bottom: 12px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_control_panel(algorithm_key):
    st.markdown("### 操作区")
    algorithm_key = st.radio(
        "选择算法",
        options=list(ALGORITHM_LABELS.keys()),
        format_func=lambda key: ALGORITHM_LABELS[key],
        key="cluster_lab_algorithm",
        label_visibility="collapsed",
    )

    dataset_key = st.selectbox(
        "选择教学数据",
        options=get_dataset_options(algorithm_key),
        format_func=get_dataset_label,
        key="cluster_lab_dataset",
    )

    col1, col2 = st.columns(2)
    if col1.button("重新生成数据", use_container_width=True):
        st.session_state["cluster_lab_seed"] += 1
    if col2.button("刷新观察点", use_container_width=True):
        st.session_state["cluster_lab_view_nonce"] += 1

    sample_cfg = sample_config(algorithm_key)
    with st.expander("数据设置", expanded=True):
        sample_count = st.slider("样本数量", sample_cfg[0], sample_cfg[1], sample_cfg[2], sample_cfg[3])
        noise = st.slider("噪声强度", 0.00, 0.80, default_noise(algorithm_key), 0.02)
        st.caption(get_dataset_summary(dataset_key))

    with st.expander("算法参数", expanded=True):
        params = render_parameter_controls(algorithm_key)

    algo_info = algorithm_overview(algorithm_key)
    st.markdown("#### 当前算法提醒")
    st.markdown(algo_info["visual_tip"])

    return {
        "algorithm_key": algorithm_key,
        "dataset_key": dataset_key,
        "sample_count": sample_count,
        "noise": noise,
        "params": params,
    }


def sample_config(algorithm_key):
    if algorithm_key == "agg":
        return (40, 110, 72, 4)
    if algorithm_key == "dbscan":
        return (100, 260, 180, 20)
    return (100, 260, 180, 20)


def default_noise(algorithm_key):
    mapping = {
        "kmeans": 0.16,
        "dbscan": 0.12,
        "agg": 0.08,
        "gmm": 0.10,
    }
    return mapping[algorithm_key]


def render_parameter_controls(algorithm_key):
    if algorithm_key == "kmeans":
        return {
            "n_clusters": st.slider("簇数量", 2, 6, 3, 1),
            "init": st.selectbox(
                "初始化方式",
                options=["kmeans++", "random"],
                format_func=lambda item: "KMeans++" if item == "kmeans++" else "随机中心",
            ),
            "n_init": st.slider("重复初始化次数", 1, 8, 4, 1),
            "max_iter": st.slider("最大迭代次数", 4, 30, 16, 1),
        }
    if algorithm_key == "dbscan":
        return {
            "eps": st.slider("邻域半径", 0.10, 0.80, 0.28, 0.01),
            "min_samples": st.slider("核心点最少邻居数", 3, 12, 5, 1),
        }
    if algorithm_key == "agg":
        return {
            "n_clusters": st.slider("簇数量", 2, 6, 3, 1),
            "linkage": st.selectbox(
                "连接方式",
                options=["single", "complete", "average", "ward"],
                format_func=lambda item: {
                    "single": "single（最近距离）",
                    "complete": "complete（最远距离）",
                    "average": "average（平均距离）",
                    "ward": "ward（方差最小化）",
                }[item],
            ),
        }
    return {
        "n_components": st.slider("高斯成分数", 2, 5, 3, 1),
        "covariance_type": st.selectbox(
            "协方差类型",
            options=["full", "diag"],
            format_func=lambda item: "full（完整协方差）" if item == "full" else "diag（对角协方差）",
        ),
        "max_iter": st.slider("最大迭代次数", 8, 36, 18, 1),
    }


def build_lab_result(algorithm_key, dataset_key, sample_count, noise, params):
    X, dataset_meta = generate_clustering_dataset(
        dataset_key=dataset_key,
        n_samples=sample_count,
        noise=noise,
        random_state=st.session_state["cluster_lab_seed"],
    )

    model = build_clusterer(algorithm_key, params)
    model.fit(X)
    labels = np.asarray(model.labels_, dtype=int)
    metrics = build_clustering_metrics(X, labels)
    extras = build_algorithm_extras(algorithm_key, dataset_key, model, X, labels, params, metrics)

    visual_context = {
        "visual_title": "{0} - {1}".format(ALGORITHM_LABELS[algorithm_key], get_dataset_label(dataset_key)),
        "X": X,
        "labels": labels,
        "metrics": metrics,
    }
    visual_context.update(extras)

    return {
        "X": X,
        "labels": labels,
        "model": model,
        "params": params,
        "metrics": metrics,
        "extras": extras,
        "visual_context": visual_context,
    }


def build_algorithm_extras(algorithm_key, dataset_key, model, X, labels, params, metrics):
    extras = {}

    if algorithm_key == "kmeans":
        extras.update(
            {
                "model": model,
                "center_history": model.center_history_,
                "initial_centers": model.initial_centers_,
                "final_centers": model.cluster_centers_,
                "iterations": model.iterations_,
                "cluster_count": metrics["cluster_count"],
                "inertia": model.inertia_,
                "suggested_clusters": dataset_suggested_clusters(dataset_key),
            }
        )
        return extras

    if algorithm_key == "dbscan":
        neighbor_sizes = np.array([len(items) for items in model.neighbor_lists_])
        core_candidates = np.where(model.core_sample_mask_)[0]
        if len(core_candidates) > 0:
            focus_index = core_candidates[np.argmax(neighbor_sizes[core_candidates])]
        else:
            focus_index = int(np.argmax(neighbor_sizes))
        extras.update(
            {
                "core_mask": model.core_sample_mask_,
                "border_mask": model.border_mask_,
                "noise_mask": model.noise_mask_,
                "eps": params["eps"],
                "min_samples": params["min_samples"],
                "focus_point": X[focus_index],
                "focus_neighbors": model.neighbor_lists_[focus_index].tolist(),
                "focus_type": point_type(model, focus_index),
                "noise_count": metrics["noise_count"],
            }
        )
        return extras

    if algorithm_key == "agg":
        extras.update(
            {
                "merge_history": model.merge_history_,
                "cut_distance": model.cut_distance_,
                "cluster_count": metrics["cluster_count"],
                "linkage": params["linkage"],
            }
        )
        return extras

    probabilities = model.responsibilities_
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-12), axis=1)
    focus_index = int(np.argmax(entropy))
    extras.update(
        {
            "model": model,
            "means": model.means_,
            "covariances": model.covariances_,
            "covariance_type": model.covariance_type,
            "weights": model.weights_,
            "focus_point": X[focus_index],
            "focus_probabilities": probabilities[focus_index],
            "focus_entropy": float(entropy[focus_index]),
            "focus_max_probability": float(np.max(probabilities[focus_index])),
        }
    )
    return extras


def point_type(model, index):
    if model.noise_mask_[index]:
        return "噪声点"
    if model.core_sample_mask_[index]:
        return "核心点"
    return "边界点"


def dataset_suggested_clusters(dataset_key):
    mapping = {
        "kmeans_blobs": 3,
        "kmeans_noisy_blobs": 4,
    }
    return mapping.get(dataset_key, 3)


def render_display_panel(algorithm_key, params, lab_result):
    algo_info = algorithm_overview(algorithm_key)
    main_col, teach_col = st.columns([1.68, 1.02], gap="large")

    with main_col:
        st.markdown("### 主图区")
        st.image(render_main_visual(algorithm_key, lab_result["visual_context"]), use_container_width=True)

    with teach_col:
        st.markdown("### 教学区")
        st.markdown('<div class="teach-note"><b>一句话速览：</b>{0}</div>'.format(algo_info["headline"]), unsafe_allow_html=True)
        tab1, tab2, tab3, tab4 = st.tabs(["原理", "参数", "现象", "适用"])
        with tab1:
            st.markdown('<div class="teach-note">{0}</div>'.format(algo_info["principle"]), unsafe_allow_html=True)
        with tab2:
            st.info(parameter_explanation(algorithm_key, params, lab_result["extras"]))
        with tab3:
            st.info(live_summary(algorithm_key, params, lab_result["metrics"], lab_result["visual_context"]))
        with tab4:
            st.markdown(
                '<div class="teach-note"><b>优点：</b>{0}<br><b>缺点：</b>{1}<br><b>适合数据：</b>{2}</div>'.format(
                    algo_info["pros"], algo_info["cons"], algo_info["fit"]
                ),
                unsafe_allow_html=True,
            )


def render_bottom_panel(algorithm_key, dataset_key, params, lab_result):
    st.markdown("### 指标与结构分析")
    metrics = lab_result["metrics"]

    metric_values = [
        ("轮廓系数", format_metric(metrics["silhouette"], "{0:.3f}")),
        ("Davies-Bouldin", format_metric(metrics["davies_bouldin"], "{0:.3f}")),
        ("Calinski-Harabasz", format_metric(metrics["calinski_harabasz"], "{0:.1f}")),
        ("当前簇数量", str(metrics["cluster_count"])),
        ("噪声点数量", str(metrics["noise_count"])),
        ("样本总数", str(len(lab_result["X"]))),
    ]

    columns = st.columns(6, gap="large")
    for column, metric in zip(columns, metric_values):
        with column:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-label">{0}</div>
                    <div class="metric-value">{1}</div>
                </div>
                """.format(metric[0], metric[1]),
                unsafe_allow_html=True,
            )

    lower_left, lower_right = st.columns([1.4, 0.9], gap="large")
    with lower_left:
        st.markdown("#### 聚类结构分析")
        st.image(render_analysis_visual(algorithm_key, lab_result["visual_context"]), use_container_width=True)

    with lower_right:
        st.markdown("#### 当前模型结论")
        st.markdown(bottom_conclusion(algorithm_key, dataset_key, metrics, lab_result["visual_context"]))


def format_metric(value, fmt):
    if value is None:
        return "--"
    return fmt.format(value)
