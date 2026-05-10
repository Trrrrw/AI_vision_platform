from .regression_data import get_dataset_label, get_dataset_summary
from .regression_models import ALGORITHM_LABELS


def regression_basics_sections():
    return [
        (
            "什么是回归任务？",
            """
            回归任务关注的是“具体数值是多少”，例如预测房价、温度、销量、考试分数。
            它的输出不是离散类别，而是连续变化的数值。
            """,
        ),
        (
            "回归和分类有什么区别？",
            """
            分类回答“属于哪一类”，回归回答“具体是多少”。
            从图像上看，分类更强调边界怎样划分，回归更强调曲线如何贴近真实趋势。
            """,
        ),
        (
            "什么是拟合与残差？",
            """
            拟合可以理解为模型画出一条线或一条曲线，去描述输入和输出之间的关系。
            残差就是“真实值减去预测值”的差，它越小说明模型越贴近数据。
            """,
        ),
        (
            "什么是欠拟合和过拟合？",
            """
            曲线太简单，抓不住真实趋势，通常叫欠拟合；
            曲线太复杂，连噪声也一起记住，通常叫过拟合。
            教学上最重要的是观察模型复杂度变化时，训练误差和测试误差如何一起变化。
            """,
        ),
        (
            "如何理解泛化能力？",
            """
            训练集上的误差反映模型是否学到了已有样本，
            测试集上的误差更能说明模型面对新数据时是否仍然可靠。
            当训练误差很低但测试误差明显变大时，往往说明模型泛化能力下降了。
            """,
        ),
    ]


def algorithm_overview(algorithm_key):
    mapping = {
        "linear": {
            "title": "线性回归",
            "headline": "用一条直线概括整体趋势，重点是理解“平均变化方向”和“残差”两个概念。",
            "principle": "线性回归假设输入和输出之间大致是线性关系，用一条最能代表整体趋势的直线去拟合数据。",
            "visual_tip": "主图会同时显示散点、拟合直线、真实趋势和若干残差线。",
            "pros": "简单、直观、适合入门教学。",
            "cons": "只能表达线性关系，对异常点较敏感。",
            "fit": "适合整体趋势接近直线的数据。",
        },
        "poly": {
            "title": "多项式回归",
            "headline": "通过提高多项式阶数得到更灵活的曲线，重点观察欠拟合和过拟合。",
            "principle": "多项式回归本质上仍是线性模型，只是把输入扩展成 x²、x³ 等高阶特征，从而得到弯曲的拟合曲线。",
            "visual_tip": "主图会随着阶数变化实时改变曲线形状，帮助学生比较“太简单”和“太复杂”的拟合。",
            "pros": "能表达明显的弯曲趋势，教学上非常适合讲复杂度。",
            "cons": "阶数太高时容易追着噪声走，测试表现反而变差。",
            "fit": "适合单峰、弯折或平滑波动的数据。",
        },
        "ridge": {
            "title": "岭回归",
            "headline": "在普通回归上加入 L2 正则化，重点观察“系数收缩”和“模型更稳”。",
            "principle": "Ridge 会惩罚过大的系数，让模型不要把权重全部压在少量特征上，因此面对相关特征时通常更稳定。",
            "visual_tip": "主图左侧是拟合曲线，右侧是系数条形图，便于同时观察曲线与权重变化。",
            "pros": "面对相关特征更稳定，能减少过拟合。",
            "cons": "会收缩系数，但通常不会把它们真正压成 0。",
            "fit": "适合特征较多、相关性较强、希望模型更稳的场景。",
        },
        "lasso": {
            "title": "套索回归",
            "headline": "在普通回归上加入 L1 正则化，重点观察“系数压缩到 0”和“特征筛选”。",
            "principle": "Lasso 会惩罚系数总和，结果往往是把部分不重要特征直接压到 0，因此特别适合讲“自动筛选特征”的想法。",
            "visual_tip": "主图右侧会显示系数条形图，很多系数会随着 alpha 增大而变短甚至消失。",
            "pros": "容易形成稀疏模型，便于解释哪些特征更重要。",
            "cons": "当特征高度相关时，选择结果可能不如 Ridge 稳定。",
            "fit": "适合有效特征较少、希望模型更简洁的场景。",
        },
        "svr": {
            "title": "支持向量回归 SVR",
            "headline": "重点观察 epsilon 容忍带、支持向量和核函数如何共同决定拟合曲线。",
            "principle": "SVR 不会执着于贴住每一个点，而是先给出一条“允许小误差”的带状区域，再重点关注落在带外的关键样本。",
            "visual_tip": "主图会显示拟合曲线、epsilon 容忍带，并高亮支持向量。",
            "pros": "适合平滑非线性拟合，对噪声有一定容忍能力。",
            "cons": "参数之间会相互影响，需要结合图像反复理解。",
            "fit": "适合非线性但仍希望拟合曲线保持平滑的场景。",
        },
        "tree": {
            "title": "决策树回归",
            "headline": "用一段一段的常数值逼近真实曲线，重点观察“阶梯状拟合”。",
            "principle": "决策树会不断切分输入区间，每个区间给出一个固定预测值，所以预测曲线看起来像一层层台阶。",
            "visual_tip": "主图会清楚展示阶梯状预测，并帮助学生理解树深度对复杂度的影响。",
            "pros": "可解释性强，适合讲分段逼近。",
            "cons": "单棵树容易跟着局部噪声摆动，稳定性有限。",
            "fit": "适合分段规律明显或局部变化突然的数据。",
        },
        "rf": {
            "title": "随机森林回归",
            "headline": "通过多棵树平均得到更稳定的预测，重点比较“单树 vs 森林”。",
            "principle": "随机森林会训练很多棵略有差异的树，再把它们的结果求平均，所以通常比单棵树更稳、更不容易被局部噪声带偏。",
            "visual_tip": "主图会并排展示单棵树与随机森林的拟合曲线，让学生直接比较差异。",
            "pros": "稳定、鲁棒、适合复杂非线性数据。",
            "cons": "比单棵树更难逐步解释内部细节。",
            "fit": "适合局部波动多、单树容易抖动的非线性数据。",
        },
    }
    return mapping[algorithm_key]


def dataset_overview(dataset_key):
    return {"title": get_dataset_label(dataset_key), "summary": get_dataset_summary(dataset_key)}


def parameter_explanation(algorithm_key, params, extras):
    if algorithm_key == "linear":
        return (
            "当前直线斜率约为 `{0:.2f}`，截距约为 `{1:.2f}`。"
            " 如果关闭偏置项，直线必须穿过原点；如果开启标准化，系数求解会更稳。"
        ).format(extras["primary_coef"], extras["intercept"])

    if algorithm_key == "poly":
        return (
            "当前多项式阶数为 `{0}`。阶数越高，曲线表达能力越强；"
            " 但当阶数过高时，曲线会开始追着噪声摆动。"
        ).format(params["degree"])

    if algorithm_key == "ridge":
        return (
            "当前 alpha = `{0:.2f}`。alpha 越大，系数收缩越明显。"
            " 这会让模型更稳，但也可能让曲线变得偏平。当前最大系数约为 `{1:.2f}`。"
        ).format(params["alpha"], extras["max_abs_coef"])

    if algorithm_key == "lasso":
        return (
            "当前 alpha = `{0:.2f}`。alpha 增大时，Lasso 会把更多系数压到接近 0。"
            " 现在共有 `{1}` 个系数接近 0，体现了特征筛选效果。"
        ).format(params["alpha"], extras["near_zero_count"])

    if algorithm_key == "svr":
        return (
            "当前 C = `{0:.2f}`，epsilon = `{1:.2f}`，kernel = `{2}`。"
            " epsilon 决定允许忽略多大的小误差，C 决定模型对带外误差有多敏感。"
        ).format(params["C"], params["epsilon"], params["kernel"])

    if algorithm_key == "tree":
        return (
            "当前最大深度为 `{0}`，最小分裂样本数为 `{1}`。"
            " 深度越大，台阶越细，越容易贴近训练样本。"
        ).format(params["max_depth"], params["min_samples_split"])

    return (
        "当前森林包含 `{0}` 棵树，最大深度为 `{1}`。"
        " 树越多，平均后的结果通常越稳定；深度越大，单棵树越容易变复杂。"
    ).format(params["n_estimators"], params["max_depth"])


def live_summary(algorithm_key, params, metrics, extras):
    gap = metrics["train_r2"] - metrics["test_r2"]

    if algorithm_key == "linear":
        return "线性回归当前更适合观察“整体趋势”和“残差”，如果真实关系明显弯曲，直线通常会显得偏简单。"

    if algorithm_key == "poly":
        if params["degree"] <= 2:
            return "当前阶数偏低，曲线更平滑，更容易欠拟合。"
        if params["degree"] >= 7:
            return "当前阶数已经偏高，曲线开始变得敏感，测试集误差更值得关注。"
        return "当前阶数处在中间范围，既有一定弯曲能力，也不至于过度追噪声。"

    if algorithm_key == "ridge":
        if params["alpha"] >= 3.0:
            return "当前正则化偏强，系数被明显收缩，模型更稳，但也可能欠拟合。"
        return "当前 Ridge 更像是在“压住过大的系数”，帮助模型减少不必要的摆动。"

    if algorithm_key == "lasso":
        if extras["near_zero_count"] >= max(2, len(extras["coef_pairs"]) // 2):
            return "当前 Lasso 已经筛掉了较多特征，模型正在变得更稀疏、更简洁。"
        return "当前 Lasso 还保留了较多特征，拟合能力较强，但筛选效果还不算明显。"

    if algorithm_key == "svr":
        if params["epsilon"] >= 0.40:
            return "当前 epsilon 较宽，模型会忽略更多小误差，曲线通常会更平滑。"
        return "当前 epsilon 较窄，模型会更在意局部误差，支持向量通常也会更多。"

    if algorithm_key == "tree":
        if params["max_depth"] >= 7:
            return "当前树深度较大，台阶会更细，更容易贴着训练样本走。"
        return "当前树深度较浅，模型更像在做粗粒度的分段逼近。"

    if gap > 0.12:
        return "训练集明显优于测试集，说明单树和森林都在变复杂，但森林通常仍会更稳一些。"
    return "随机森林正在通过多棵树平均削弱单树的局部抖动，这是它最大的教学重点。"


def bottom_conclusion(algorithm_key, dataset_key, metrics, extras):
    gap = metrics["train_r2"] - metrics["test_r2"]
    if gap > 0.15:
        conclusion = "训练集表现明显好于测试集，当前参数组合偏向过拟合。"
    elif gap > 0.06:
        conclusion = "训练与测试之间有一定差距，模型复杂度开始提升。"
    else:
        conclusion = "训练集与测试集误差差距不大，当前拟合相对稳定。"

    detail = ""
    if algorithm_key == "lasso":
        detail = " 当前约有 {0} 个系数接近 0，稀疏性比较明显。".format(extras["near_zero_count"])
    elif algorithm_key == "svr":
        detail = " 当前支持向量数量为 {0}。".format(extras["support_vector_count"])
    elif algorithm_key == "rf":
        detail = " 随机森林相对单棵树的测试 R² 提升约 {0:.2f}。".format(extras["forest_gain"])

    return """
- 当前算法：`{0}`
- 当前数据集：`{1}`
- 测试集 R²：`{2:.3f}`
- 测试集 RMSE：`{3:.3f}`
- 教学结论：{4}{5}
""".format(
        ALGORITHM_LABELS[algorithm_key],
        get_dataset_label(dataset_key),
        metrics["test_r2"],
        metrics["test_rmse"],
        conclusion,
        detail,
    )
