from pages_modules.classification_data_generation import get_dataset_label, get_dataset_summary
from .classification_model_factory import ALGORITHM_LABELS


def classification_basics_sections():
    return [
        (
            "分类任务是什么？",
            """
            分类任务就是让模型学会把样本分到正确类别里。
            例如判断邮件是不是垃圾邮件、图片里是猫还是狗、病人是否患病，
            输出通常是离散标签，而不是连续数值。
            """,
        ),
        (
            "分类和回归的区别",
            """
            分类关注“属于哪一类”，回归关注“具体是多少”。
            从图像上看，分类更重视边界如何划分，回归更重视曲线如何拟合。
            """,
        ),
        (
            "什么是决策边界？",
            """
            决策边界可以理解为模型在平面里画出的分界线。
            边界越弯曲，模型通常越灵活；边界越平滑，模型通常越简单。
            观察边界变化，是理解分类算法差异最直接的方法。
            """,
        ),
        (
            "训练集、测试集与泛化能力",
            """
            训练集用于学习规律，测试集用于检查模型在新样本上的表现。
            如果训练准确率很高，但测试准确率明显偏低，往往说明模型过拟合了。
            """,
        ),
    ]


def algorithm_overview(algorithm_key):
    mapping = {
        "knn": {
            "title": "KNN（K-近邻）",
            "headline": "通过最近邻投票完成分类，重点是理解“谁离我近，谁就更有发言权”。",
            "principle": """
            KNN 的思路非常直观：一个新样本大概率属于它附近样本占多数的那个类别。
            所以它最适合讲解局部结构、邻域投票和边界平滑程度。
            """,
            "visual_tip": "主图会额外高亮一个查询点，并显示它最近的 K 个邻居及投票结果。",
            "pros": "直观、适合教学、能明显展示欠拟合和过拟合。",
            "cons": "预测时依赖全体训练样本，数据量大时会变慢，对噪声敏感。",
            "fit": "适合边界依赖局部邻域结构的数据，比如双月牙、同心圆。",
        },
        "svm": {
            "title": "SVM（支持向量机）",
            "headline": "通过最大化分类间隔来划分样本，重点是理解“最大间隔”和“支持向量”。",
            "principle": """
            SVM 会寻找一条最能分开不同类别的边界，并尽量让边界两侧留出更大的安全间隔。
            当线性边界不够用时，核函数会把数据映射到更容易分开的空间。
            """,
            "visual_tip": "主图会高亮支持向量，并把决策边界与间隔带分开显示。",
            "pros": "边界清晰，适合讲解间隔、支持向量和核函数。",
            "cons": "参数之间会相互影响，初学者需要结合图像反复理解。",
            "fit": "适合近线性可分或具有清晰结构的数据，也适合展示核函数效果。",
        },
        "nb": {
            "title": "朴素贝叶斯",
            "headline": "通过比较类别概率完成分类，重点是理解“概率区域”和“条件独立假设”。",
            "principle": """
            朴素贝叶斯会计算样本分别属于各类别的概率，再选出概率最大的那个类别。
            它做了一个重要简化：默认特征在给定类别条件下近似独立。
            """,
            "visual_tip": "主图会用颜色深浅表示某一区域对某个类别的置信度高低。",
            "pros": "训练快、解释性强、特别适合讲概率分类思想。",
            "cons": "独立性假设较强，面对复杂边界时灵活性不够。",
            "fit": "适合特征近似独立、强调概率推断而不是几何分割的任务。",
        },
        "rf": {
            "title": "随机森林",
            "headline": "通过多棵树投票来得到更稳定的结果，重点是理解“单树”和“森林”的差异。",
            "principle": """
            随机森林会训练很多棵决策树，每棵树看到的样本和特征都略有不同，
            最后通过投票给出最终分类结果，因此通常比单棵树更稳定。
            """,
            "visual_tip": "主图会同时展示单棵树和随机森林的分类区域，帮助学生理解集成思想。",
            "pros": "稳定、鲁棒、适合展示复杂局部边界和集成学习。",
            "cons": "可解释性不如单棵树直观，树太多时计算量会增加。",
            "fit": "适合分块结构明显、局部边界复杂、需要观察稳定性变化的数据。",
        },
    }
    return mapping[algorithm_key]


def dataset_overview(dataset_key):
    return {
        "title": get_dataset_label(dataset_key),
        "summary": get_dataset_summary(dataset_key),
    }


def parameter_explanation(algorithm_key, params, extras):
    if algorithm_key == "knn":
        prediction = extras["query_pred"]
        vote_text = "，".join("类别 {0}: {1:.2f}".format(label, score) for label, score in extras["vote_scores"].items())
        return """
        当前 `K = {0}`，投票方式是 `{1}`。
        这意味着查询点会参考最近的 {0} 个邻居来决定类别。
        目前这个示例查询点的投票情况为：{2}，因此它被判为 `类别 {3}`。
        """.format(
            params["n_neighbors"],
            "距离加权" if params["weight_mode"] == "distance" else "均匀投票",
            vote_text,
            prediction,
        )

    if algorithm_key == "svm":
        margin = extras.get("margin_width")
        margin_text = "当前线性间隔宽度约为 {0:.2f}。".format(margin) if margin is not None else "当前核函数下更适合观察边界形状，而不是精确间隔宽度。"
        return """
        当前 `C = {0:.1f}`，核函数为 `{1}`，`gamma = {2:.1f}`。
        {3}
        如果 C 增大，模型会更重视训练集拟合；如果 gamma 增大，边界通常会更弯曲。
        """.format(params["C"], params["kernel"], params["gamma"], margin_text)

    if algorithm_key == "nb":
        probability = extras["query_proba"]
        return """
        当前 `alpha = {0:.3f}`。
        对示例查询点而言，模型认为它属于 `类别 0` 的概率约为 {1:.1f}%，属于 `类别 1` 的概率约为 {2:.1f}%。
        朴素贝叶斯并不是在画几何边界，而是在比较不同类别的概率大小。
        """.format(params["alpha"], probability[0] * 100, probability[1] * 100)

    return """
    当前森林包含 `{0}` 棵树，最大深度为 `{1}`，最小分裂样本数为 `{2}`。
    单棵树通常边界更碎、更容易受样本扰动影响；森林投票后边界会更稳定。
    当前测试集中，单棵树与森林有 `{3}` 个样本给出了不同预测。
    """.format(
        params["n_estimators"],
        params["max_depth"],
        params["min_samples_split"],
        extras["tree_forest_disagree"],
    )


def live_summary(algorithm_key, params):
    if algorithm_key == "knn":
        if params["n_neighbors"] <= 3:
            return "当前 K 值偏小，边界会更贴着训练样本走，更容易受噪声影响。"
        if params["n_neighbors"] >= 12:
            return "当前 K 值偏大，边界会明显更平滑，但局部细节可能被抹掉。"
        return "当前 K 值处于中等范围，既保留了一定局部结构，又不至于太敏感。"

    if algorithm_key == "svm":
        if params["kernel"] == "linear":
            return "当前正在强调线性可分与最大间隔思想，最适合观察支持向量和 margin。"
        return "当前核函数会让边界拥有更强的弯曲能力，适合观察非线性分类。"

    if algorithm_key == "nb":
        if params["alpha"] < 0.05:
            return "平滑参数较小，模型更相信原始概率分布，边界会更贴近样本分布。"
        return "平滑参数较大，模型会更保守，概率区域变化会更平缓。"

    if params["max_depth"] >= 7:
        return "当前树深度较大，单棵树会更复杂，森林虽然更稳，但仍可能开始过拟合。"
    return "当前森林更偏向稳健划分，适合观察集成学习如何平滑单棵树的波动。"


def bottom_conclusion(algorithm_key, dataset_key, train_acc, test_acc, misclassified_count):
    gap = train_acc - test_acc
    if gap > 0.12:
        generalization = "训练准确率明显高于测试准确率，当前参数组合偏向过拟合。"
    elif gap > 0.05:
        generalization = "训练集略优于测试集，模型开始呈现一定复杂度优势。"
    else:
        generalization = "训练集与测试集比较接近，当前模型泛化表现较稳定。"

    return """
    - 当前算法：`{0}`
    - 当前教学数据：`{1}`
    - 错分测试样本数：`{2}`
    - 教学结论：{3}
    """.format(ALGORITHM_LABELS[algorithm_key], get_dataset_label(dataset_key), misclassified_count, generalization)


def home_learning_cards():
    return [
        ("机器学习", "从分类、回归、聚类出发，通过图像与交互理解经典算法。"),
        ("神经网络", "逐步理解前馈网络、卷积网络、循环网络和注意力机制。"),
        ("国产大模型", "聚焦 DeepSeek、智谱、通义千问等模型的能力与应用。"),
    ]
