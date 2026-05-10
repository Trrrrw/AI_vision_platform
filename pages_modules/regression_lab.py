import streamlit as st
import numpy as np

from pages_modules.regression_data import (
    build_plot_features,
    generate_regression_dataset,
    get_dataset_label,
    get_dataset_options,
    get_dataset_summary,
    get_default_dataset,
    reference_signal,
    split_indices,
)
from pages_modules.regression_metrics import build_regression_metrics
from pages_modules.regression_models import ALGORITHM_LABELS, build_regressor
from pages_modules.regression_text import (
    algorithm_overview,
    bottom_conclusion,
    dataset_overview,
    live_summary,
    parameter_explanation,
    regression_basics_sections,
)
from pages_modules.regression_viz import render_diagnostic_visual, render_main_visual


STATE_DEFAULTS = {
    "reg_lab_algorithm": "linear",
    "reg_lab_dataset": "linear_trend",
    "reg_lab_seed": 13,
    "reg_lab_view_nonce": 0,
    "reg_lab_prev_algorithm": "linear",
}


def render_regression_lab():
    ensure_state()
    sync_dataset_with_algorithm()
    inject_page_css()

    algorithm_key = st.session_state["reg_lab_algorithm"]
    dataset_key = st.session_state["reg_lab_dataset"]
    algo_info = algorithm_overview(algorithm_key)
    dataset_info = dataset_overview(dataset_key)

    st.markdown(
        """
        <div class="lab-hero">
            <div class="lab-overline">机器学习 · 回归</div>
            <div class="lab-title">回归算法可视化学习实验室</div>
            <div class="lab-subtitle">
                先选算法，再切换对应教学数据，随后观察拟合曲线、残差和误差指标如何一起变化。
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

    with st.expander("回归基础知识", expanded=False):
        for index, (title, body) in enumerate(regression_basics_sections()):
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
        test_size=settings["test_size"],
        params=settings["params"],
    )

    with content_col:
        render_display_panel(settings["algorithm_key"], settings["dataset_key"], settings["params"], lab_result)
        st.markdown("<div style='height: 0.4rem;'></div>", unsafe_allow_html=True)
        render_bottom_panel(settings["algorithm_key"], settings["dataset_key"], settings["params"], lab_result)


def ensure_state():
    for key, value in STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sync_dataset_with_algorithm():
    current_algorithm = st.session_state["reg_lab_algorithm"]
    previous_algorithm = st.session_state["reg_lab_prev_algorithm"]
    valid_options = get_dataset_options(current_algorithm)
    if previous_algorithm != current_algorithm or st.session_state["reg_lab_dataset"] not in valid_options:
        st.session_state["reg_lab_dataset"] = get_default_dataset(current_algorithm)
        st.session_state["reg_lab_prev_algorithm"] = current_algorithm
        st.session_state["reg_lab_view_nonce"] += 1


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
            .panel-card {
                background: #ffffff;
                border: 1px solid #e4eef7;
                border-radius: 20px;
                padding: 20px;
                box-shadow: 0 8px 20px rgba(15, 91, 158, 0.05);
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
        key="reg_lab_algorithm",
        label_visibility="collapsed",
    )

    valid_datasets = get_dataset_options(algorithm_key)
    dataset_key = st.selectbox(
        "选择教学数据",
        options=valid_datasets,
        format_func=get_dataset_label,
        key="reg_lab_dataset",
    )

    col1, col2 = st.columns(2)
    if col1.button("重新生成数据", use_container_width=True):
        st.session_state["reg_lab_seed"] += 1
    if col2.button("刷新观察点", use_container_width=True):
        st.session_state["reg_lab_view_nonce"] += 1

    with st.expander("数据设置", expanded=True):
        sample_count = st.slider("样本数量", 100, 320, 180, 20)
        noise = st.slider("噪声强度", 0.00, 0.80, default_noise(algorithm_key), 0.02)
        test_size = st.slider("测试集比例", 0.20, 0.40, 0.30, 0.02)
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
    if algorithm_key == "linear":
        return {
            "fit_intercept": st.checkbox("加入偏置项", value=True),
            "standardize": st.checkbox("先标准化特征", value=True),
        }
    if algorithm_key == "poly":
        return {"degree": st.slider("多项式阶数", 2, 9, 4, 1)}
    if algorithm_key == "ridge":
        return {"alpha": st.slider("正则化强度", 0.10, 4.00, 1.20, 0.05)}
    if algorithm_key == "lasso":
        return {"alpha": st.slider("正则化强度", 0.01, 1.00, 0.12, 0.01)}
    if algorithm_key == "svr":
        return {
            "C": st.slider("惩罚系数 C", 0.2, 4.0, 1.4, 0.1),
            "epsilon": st.slider("容忍带宽度", 0.05, 0.60, 0.22, 0.01),
            "kernel": st.selectbox(
                "核函数",
                options=["linear", "rbf", "poly"],
                format_func=lambda item: {"linear": "线性核", "rbf": "RBF 核", "poly": "多项式核"}[item],
            ),
            "gamma": st.slider("核函数影响范围", 0.10, 2.50, 0.80, 0.05),
        }
    if algorithm_key == "tree":
        return {
            "max_depth": st.slider("最大深度", 1, 10, 4, 1),
            "min_samples_split": st.slider("分裂所需最少样本数", 2, 20, 6, 1),
        }
    return {
        "n_estimators": st.slider("树的数量", 5, 41, 21, 2),
        "max_depth": st.slider("最大深度", 2, 10, 5, 1),
        "min_samples_split": st.slider("分裂所需最少样本数", 2, 20, 6, 1),
    }


def default_noise(algorithm_key):
    mapping = {
        "linear": 0.18,
        "poly": 0.14,
        "ridge": 0.18,
        "lasso": 0.16,
        "svr": 0.14,
        "tree": 0.10,
        "rf": 0.16,
    }
    return mapping[algorithm_key]


def build_lab_result(algorithm_key, dataset_key, sample_count, noise, test_size, params):
    X, y, dataset_meta = generate_regression_dataset(
        dataset_key=dataset_key,
        n_samples=sample_count,
        noise=noise,
        random_state=st.session_state["reg_lab_seed"],
    )
    x_all = dataset_meta["plot_x"]
    feature_names = dataset_meta["feature_names"]

    train_index, test_index = split_indices(len(X), test_size, st.session_state["reg_lab_seed"])
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    x_train, x_test = x_all[train_index], x_all[test_index]

    model = build_regressor(algorithm_key, params)
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    x_grid = np.linspace(x_all.min() - 0.15, x_all.max() + 0.15, 280)
    X_grid = build_plot_features(dataset_key, x_grid)
    y_grid_pred = model.predict(X_grid)
    y_grid_true = reference_signal(dataset_key, x_grid)

    metrics = build_regression_metrics(y_train, y_train_pred, y_test, y_test_pred)
    focus_index = st.session_state["reg_lab_view_nonce"] % len(X_test)
    residual_demo_indices = np.argsort(np.abs(y_test_pred - y_test))[::-1]
    x_test_sort_index = np.argsort(x_test)

    visual_context = {
        "visual_title": "{0}｜{1}".format(ALGORITHM_LABELS[algorithm_key], get_dataset_label(dataset_key)),
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
        "x_grid": x_grid,
        "y_grid_pred": y_grid_pred,
        "y_grid_true": y_grid_true,
        "focus_x": float(x_test[focus_index]),
        "focus_true": float(y_test[focus_index]),
        "focus_pred": float(y_test_pred[focus_index]),
        "focus_residual": float(y_test_pred[focus_index] - y_test[focus_index]),
        "residual_demo_indices": residual_demo_indices.tolist(),
        "x_test_sorted": x_test[x_test_sort_index],
        "x_test_sort_index": x_test_sort_index,
        "y_test_sorted": y_test[x_test_sort_index],
        "y_test_pred_sorted": y_test_pred[x_test_sort_index],
        "metrics": metrics,
    }

    extras = build_algorithm_extras(
        algorithm_key=algorithm_key,
        params=params,
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        y_train_pred=y_train_pred,
        y_test_pred=y_test_pred,
        x_grid=x_grid,
        feature_names=feature_names,
        dataset_key=dataset_key,
    )
    if algorithm_key == "rf":
        extras["forest_gain"] = float(metrics["test_r2"] - extras["single_tree_metrics"]["test_r2"])
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
        "x_train": x_train,
        "x_test": x_test,
        "model": model,
        "params": params,
        "metrics": metrics,
        "extras": extras,
        "visual_context": visual_context,
        "dataset_key": dataset_key,
        "algorithm_key": algorithm_key,
    }


def build_algorithm_extras(
    algorithm_key,
    params,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    y_train_pred,
    y_test_pred,
    x_grid,
    feature_names,
    dataset_key,
):
    extras = {}

    if algorithm_key == "linear":
        extras["primary_coef"] = float(model.coef_[0])
        extras["intercept"] = float(model.intercept_)
        return extras

    if algorithm_key == "poly":
        extras["degree"] = params["degree"]
        if params["degree"] <= 2:
            extras["shape_note"] = "曲线目前更偏平滑，适合观察欠拟合。"
        elif params["degree"] >= 7:
            extras["shape_note"] = "曲线弯折已经比较明显，要特别留意测试集误差。"
        else:
            extras["shape_note"] = "曲线复杂度处于中间水平，通常更适合教学比较。"
        return extras

    if algorithm_key in ["ridge", "lasso"]:
        coef_pairs = list(zip(feature_names, model.coef_.tolist()))
        extras["coef_pairs"] = coef_pairs
        extras["alpha"] = params["alpha"]
        extras["max_abs_coef"] = float(max(abs(value) for _, value in coef_pairs))
        extras["near_zero_count"] = int(sum(abs(value) < 0.08 for _, value in coef_pairs))
        return extras

    if algorithm_key == "svr":
        support_index = model.support_indices()
        train_residual = np.abs(y_train_pred - y_train)
        extras["support_x"] = X_train[support_index, 0]
        extras["support_y"] = y_train[support_index]
        extras["support_vector_count"] = model.support_vector_count()
        extras["epsilon"] = params["epsilon"]
        extras["tube_ratio"] = float(np.mean(train_residual <= params["epsilon"]))
        return extras

    if algorithm_key == "tree":
        grid_prediction = model.predict(build_plot_features(dataset_key, x_grid))
        extras["leaf_count"] = int(model.leaf_count_)
        extras["plateau_count"] = int(len(np.unique(np.round(grid_prediction, 2))))
        return extras

    single_tree_train_pred = model.single_tree_predict(X_train)
    single_tree_test_pred = model.single_tree_predict(X_test)
    single_tree_metrics = build_regression_metrics(y_train, single_tree_train_pred, y_test, single_tree_test_pred)
    extras["single_tree_grid_pred"] = model.single_tree_predict(build_plot_features(dataset_key, x_grid))
    extras["single_tree_metrics"] = single_tree_metrics
    extras["single_tree_test_sorted"] = single_tree_test_pred[np.argsort(X_test[:, 0])]
    return extras


def render_display_panel(algorithm_key, dataset_key, params, lab_result):
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
            st.info(live_summary(algorithm_key, params, lab_result["metrics"], lab_result["extras"]))
        with tab4:
            st.markdown(
                '<div class="teach-note"><b>优点：</b>{0}<br><b>缺点：</b>{1}<br><b>适合数据：</b>{2}</div>'.format(
                    algo_info["pros"], algo_info["cons"], algo_info["fit"]
                ),
                unsafe_allow_html=True,
            )


def render_bottom_panel(algorithm_key, dataset_key, params, lab_result):
    st.markdown("### 结果与误差分析")
    metrics = lab_result["metrics"]
    metric_values = [
        ("训练集 R²", "{0:.3f}".format(metrics["train_r2"])),
        ("测试集 R²", "{0:.3f}".format(metrics["test_r2"])),
        ("测试集 MSE", "{0:.3f}".format(metrics["test_mse"])),
        ("测试集 MAE", "{0:.3f}".format(metrics["test_mae"])),
        ("测试集 RMSE", "{0:.3f}".format(metrics["test_rmse"])),
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
        st.markdown("#### 残差与预测关系")
        st.image(render_diagnostic_visual(algorithm_key, lab_result["visual_context"]), use_container_width=True)

    with lower_right:
        st.markdown("#### 当前模型结论")
        st.markdown(bottom_conclusion(algorithm_key, dataset_key, metrics, lab_result["extras"]))
