import numpy as np
import streamlit as st

from pages_modules.classification_data_generation import (
    StandardScaler,
    class_names_from_labels,
    generate_algorithm_dataset,
    get_dataset_label,
    get_dataset_options,
    get_dataset_summary,
    get_default_dataset,
    train_test_split,
)
from pages_modules.classification_metrics import accuracy_score, confusion_matrix, misclassified_count
from pages_modules.classification_model_factory import ALGORITHM_LABELS, build_classifier
from pages_modules.classification_teaching_text import (
    algorithm_overview,
    bottom_conclusion,
    classification_basics_sections,
    dataset_overview,
    live_summary,
    parameter_explanation,
)
from pages_modules.classification_visualization import CLASS_COLORS, hex_to_rgb, render_confusion_matrix_image, render_main_visual


STATE_DEFAULTS = {
    "cls_lab_algorithm": "knn",
    "cls_lab_dataset": "knn_moons",
    "cls_lab_seed": 21,
    "cls_lab_view_nonce": 0,
    "cls_lab_prev_algorithm": "knn",
}


def render_classification_lab():
    ensure_state()
    sync_dataset_with_algorithm()
    inject_page_css()

    algorithm_key = st.session_state["cls_lab_algorithm"]
    dataset_key = st.session_state["cls_lab_dataset"]
    algo_info = algorithm_overview(algorithm_key)
    dataset_info = dataset_overview(dataset_key)

    st.markdown(
        '<a href="/" target="_self" style="text-decoration:none; font-size:14px; color:#1A7EC1;">🏠 返回首页</a>',
        unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        """
        <div class="lab-hero">
            <div class="lab-overline">机器学习 · 分类</div>
            <div class="lab-title">分类算法可视化学习实验室</div>
            <div class="lab-subtitle">先选算法，再切换专属教学数据，再观察图像、参数和结论如何一起变化。</div>
            <div class="lab-badges">
                <span class="lab-badge">当前算法：{0}</span>
                <span class="lab-badge">当前数据：{1}</span>
            </div>
            <div class="lab-summary-grid">
                <div class="lab-summary-card"><b>算法一句话：</b>{2}</div>
                <div class="lab-summary-card"><b>数据一句话：</b>{3}</div>
            </div>
        </div>
        """.format(
            algo_info["title"],
            dataset_info["title"],
            algo_info["headline"],
            dataset_info["summary"],
        ),
        unsafe_allow_html=True,
    )

    with st.expander("分类基础知识", expanded=False):
        for index, (title, body) in enumerate(classification_basics_sections()):
            st.markdown("**{0}. {1}**".format(index + 1, title))
            st.markdown(body)

    control_col, display_col = st.columns([0.9, 2.2], gap="large")
    with control_col:
        settings = render_control_panel(algorithm_key)

    lab_result = build_lab_result(
        algorithm_key=algorithm_key,
        dataset_key=settings["dataset_key"],
        sample_count=settings["sample_count"],
        noise=settings["noise"],
        test_size=settings["test_size"],
        params=settings["params"],
    )

    with display_col:
        render_display_panel(algorithm_key, settings["params"], lab_result)

    render_bottom_panel(algorithm_key, settings["params"], lab_result)


def ensure_state():
    for key, value in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sync_dataset_with_algorithm():
    current_algorithm = st.session_state["cls_lab_algorithm"]
    previous_algorithm = st.session_state["cls_lab_prev_algorithm"]
    valid_options = get_dataset_options(current_algorithm)
    if previous_algorithm != current_algorithm or st.session_state["cls_lab_dataset"] not in valid_options:
        st.session_state["cls_lab_dataset"] = get_default_dataset(current_algorithm)
        st.session_state["cls_lab_prev_algorithm"] = current_algorithm
        st.session_state["cls_lab_view_nonce"] += 1


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
                background: rgba(255,255,255,0.85);
                border: 1px solid #e1edf7;
                border-radius: 18px;
                padding: 14px 16px;
                color: #4f6475;
                line-height: 1.8;
            }
            .panel-card {
                background: #ffffff;
                border: 1px solid #e4eef7;
                border-radius: 20px;
                padding: 20px 20px;
                box-shadow: 0 8px 20px rgba(15, 91, 158, 0.05);
            }
            .metric-card {
                background: #ffffff;
                border: 1px solid #e4eef7;
                border-radius: 18px;
                padding: 16px 18px;
                box-shadow: 0 8px 18px rgba(15, 91, 158, 0.05);
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
        key="cls_lab_algorithm",
        label_visibility="collapsed",
    )

    valid_datasets = get_dataset_options(algorithm_key)
    dataset_key = st.selectbox(
        "选择教学数据",
        options=valid_datasets,
        format_func=get_dataset_label,
        key="cls_lab_dataset",
    )

    col1, col2 = st.columns(2)
    if col1.button("重新生成数据", use_container_width=True):
        st.session_state["cls_lab_seed"] += 1
    if col2.button("刷新示例视角", use_container_width=True):
        st.session_state["cls_lab_view_nonce"] += 1

    with st.expander("数据设置", expanded=True):
        sample_count = st.slider("样本数量", 120, 320, 220, 20)
        noise = st.slider("噪声强度", 0.00, 0.50, default_noise(algorithm_key), 0.02)
        test_size = st.slider("测试集比例", 0.20, 0.40, 0.28, 0.02)
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
        "test_size": test_size,
        "params": params,
    }


def render_parameter_controls(algorithm_key):
    if algorithm_key == "knn":
        return {
            "n_neighbors": st.slider("邻居数量", 1, 25, 5, 1),
            "weight_mode": st.selectbox(
                "投票方式",
                options=["uniform", "distance"],
                format_func=lambda mode: "均匀投票" if mode == "uniform" else "距离加权",
            ),
        }
    if algorithm_key == "svm":
        return {
            "C": st.slider("惩罚系数 C", 0.1, 8.0, 1.2, 0.1),
            "kernel": st.selectbox(
                "核函数",
                options=["linear", "rbf", "poly"],
                format_func=lambda item: {"linear": "线性核", "rbf": "RBF 核", "poly": "多项式核"}[item],
            ),
            "gamma": st.slider("核函数影响范围", 0.1, 4.0, 1.0, 0.1),
        }
    if algorithm_key == "nb":
        return {
            "alpha": st.slider("平滑参数", 0.001, 1.000, 0.060, 0.001),
        }
    return {
        "n_estimators": st.slider("树的数量", 3, 31, 11, 2),
        "max_depth": st.slider("最大深度", 1, 10, 4, 1),
        "min_samples_split": st.slider("分裂所需最少样本数", 2, 20, 4, 1),
    }


def default_noise(algorithm_key):
    mapping = {
        "knn": 0.18,
        "svm": 0.16,
        "nb": 0.12,
        "rf": 0.14,
    }
    return mapping[algorithm_key]


def build_lab_result(algorithm_key, dataset_key, sample_count, noise, test_size, params):
    X, y = generate_algorithm_dataset(
        dataset_key=dataset_key,
        n_samples=sample_count,
        noise=noise,
        random_state=st.session_state["cls_lab_seed"],
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=st.session_state["cls_lab_seed"],
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = build_classifier(algorithm_key, params)
    model.fit(X_train_scaled, y_train)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    extras = build_algorithm_extras(
        algorithm_key=algorithm_key,
        model=model,
        scaler=scaler,
        X_train=X_train,
        X_train_scaled=X_train_scaled,
        y_train=y_train,
        X_test=X_test,
        X_test_scaled=X_test_scaled,
        y_test=y_test,
        y_test_pred=y_test_pred,
    )

    visual_context = {
        "model": model,
        "scaler": scaler,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "visual_title": "{0} · {1}".format(ALGORITHM_LABELS[algorithm_key], get_dataset_label(dataset_key)),
    }
    visual_context.update(extras)

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "model": model,
        "scaler": scaler,
        "params": params,
        "extras": extras,
        "visual_context": visual_context,
        "train_acc": accuracy_score(y_train, y_train_pred),
        "test_acc": accuracy_score(y_test, y_test_pred),
        "confusion": confusion_matrix(y_test, y_test_pred),
        "misclassified": misclassified_count(y_test, y_test_pred),
        "class_names": class_names_from_labels(y),
    }


def build_algorithm_extras(algorithm_key, model, scaler, X_train, X_train_scaled, y_train, X_test, X_test_scaled, y_test, y_test_pred):
    nonce = st.session_state["cls_lab_view_nonce"]
    sample_index = nonce % len(X_test)
    query_point = X_test[sample_index]
    query_scaled = X_test_scaled[sample_index]

    if algorithm_key == "knn":
        report = model.get_neighbor_report(query_scaled)
        return {
            "query_point": query_point,
            "query_true": int(y_test[sample_index]),
            "query_pred": int(report["predicted_label"]),
            "neighbor_points": X_train[report["indices"]],
            "vote_scores": report["vote_scores"],
        }

    if algorithm_key == "svm":
        support_vectors_scaled = model.support_vectors()
        margin_width = model.linear_margin_width()
        margin_note = (
            "当前线性间隔宽度约为 {0:.2f}。".format(margin_width)
            if margin_width is not None
            else "当前核函数更适合观察边界形状变化，而不是固定间隔宽度。"
        )
        return {
            "support_vectors": support_vectors_scaled,
            "support_vector_count": model.support_vector_count(),
            "margin_width": margin_width,
            "margin_note": margin_note,
        }

    if algorithm_key == "nb":
        query_proba = model.predict_proba(query_scaled.reshape(1, -1))[0]
        distribution_boxes = []
        for class_id in model.classes_:
            center_scaled = model.theta_[class_id]
            center = scaler.inverse_transform(center_scaled)
            std = np.sqrt(model.var_[class_id]) * scaler.scale_
            distribution_boxes.append(
                {
                    "center": center,
                    "std": std,
                    "color": hex_to_rgb(CLASS_COLORS[int(class_id)]),
                }
            )
        return {
            "query_point": query_point,
            "query_proba": query_proba,
            "distribution_boxes": distribution_boxes,
        }

    single_tree_pred = model.single_tree_predict(X_test_scaled)
    return {
        "tree_forest_disagree": int((single_tree_pred != y_test_pred).sum()),
    }


def render_display_panel(algorithm_key, params, lab_result):
    st.markdown("### 展示区")
    st.image(render_main_visual(algorithm_key, lab_result["visual_context"]), use_container_width=True)

    summary_col1, summary_col2 = st.columns(2, gap="large")
    with summary_col1:
        st.markdown("#### 当前图像为什么会这样")
        st.info(live_summary(algorithm_key, params))
    with summary_col2:
        st.markdown("#### 当前参数意味着什么")
        st.info(parameter_explanation(algorithm_key, params, lab_result["extras"]))


def render_bottom_panel(algorithm_key, params, lab_result):
    st.markdown("### 结果与教学解释")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4, gap="large")
    metrics = [
        ("训练准确率", "{0:.1f}%".format(lab_result["train_acc"] * 100)),
        ("测试准确率", "{0:.1f}%".format(lab_result["test_acc"] * 100)),
        ("样本总数", str(len(lab_result["X"]))),
        ("错分测试样本", str(lab_result["misclassified"])),
    ]
    for col, metric in zip([metric_col1, metric_col2, metric_col3, metric_col4], metrics):
        label, value = metric
        with col:
            st.markdown(
                """
                <div class="metric-card">
                    <div class="metric-label">{0}</div>
                    <div class="metric-value">{1}</div>
                </div>
                """.format(label, value),
                unsafe_allow_html=True,
            )

    lower_left, lower_right = st.columns([1.05, 1.2], gap="large")
    with lower_left:
        st.markdown("#### 混淆矩阵")
        st.image(
            render_confusion_matrix_image(lab_result["confusion"], lab_result["class_names"], "测试集混淆矩阵"),
            use_container_width=True,
        )
        st.markdown(bottom_conclusion(algorithm_key, st.session_state["cls_lab_dataset"], lab_result["train_acc"], lab_result["test_acc"], lab_result["misclassified"]))

    with lower_right:
        info = algorithm_overview(algorithm_key)
        st.markdown("#### 算法原理")
        st.markdown(info["principle"])
        st.markdown("#### 优点 / 局限")
        st.markdown("- 优点：{0}".format(info["pros"]))
        st.markdown("- 局限：{0}".format(info["cons"]))
        st.markdown("#### 适合什么类型的数据")
        st.markdown(info["fit"])
