from .clustering_data import get_dataset_label, get_dataset_summary
from .clustering_models import ALGORITHM_LABELS


def clustering_basics_sections():
    return [
        (
            "什么是聚类？",
            """
            聚类就是把“彼此相似”的样本自动分成一组。
            它不需要老师事先告诉模型每个点属于哪一类，而是让算法自己去发现数据中的结构。
            """,
        ),
        (
            "聚类和分类有什么区别？",
            """
            分类是有监督学习，训练时会给出真实标签；
            聚类是无监督学习，算法只能根据样本之间的距离、密度或分布关系自己分组。
            """,
        ),
        (
            "为什么说聚类没有标签？",
            """
            在聚类任务里，输入通常只有样本本身，没有“正确答案”告诉模型该怎么分。
            所以聚类结果好不好，更多依赖数据结构和评价指标，而不是看有没有预测对标签。
            """,
        ),
        (
            "什么叫簇内相似、簇间差异？",
            """
            理想情况下，同一个簇里的点应该彼此更接近，不同簇之间应该更分开。
            这也是轮廓系数、Davies-Bouldin 等指标背后想表达的核心思想。
            """,
        ),
        (
            "为什么数据形状会强烈影响聚类结果？",
            """
            有的算法偏好圆形簇，有的算法擅长找任意形状簇，有的算法更适合概率分布或层次结构。
            所以同一批数据换一种算法，结果可能完全不同。
            """,
        ),
    ]


def algorithm_overview(algorithm_key):
    mapping = {
        "kmeans": {
            "title": "KMeans 聚类",
            "headline": "通过寻找若干簇中心，把点分给离自己最近的中心，重点看“中心移动”和“簇内平方误差”。",
            "principle": "KMeans 会先随机选出若干中心，再反复执行“按最近中心分组”和“更新中心位置”这两步，直到中心稳定下来。",
            "visual_tip": "主图会同时显示初始中心、最终中心和中心移动轨迹，让学生看到迭代是如何收敛的。",
            "pros": "直观、速度快、特别适合规则球状簇。",
            "cons": "对初始中心、簇数和异常点比较敏感，也不擅长任意形状簇。",
            "fit": "适合中心明显、形状接近圆形的簇。",
        },
        "dbscan": {
            "title": "DBSCAN 密度聚类",
            "headline": "通过密度把点连接成簇，重点看“核心点、边界点、噪声点”和“eps 邻域”。",
            "principle": "DBSCAN 会先找出周围邻居足够多的核心点，再从核心点向外扩展，把密度可达的点连成同一个簇。",
            "visual_tip": "主图会区分核心点、边界点和噪声点，并高亮一个 eps 邻域示例。",
            "pros": "能处理任意形状簇，还能自动识别噪声。",
            "cons": "当不同簇密度差别很大时，参数就会比较难调。",
            "fit": "适合弯曲簇、环形簇、带噪声的二维点集。",
        },
        "agg": {
            "title": "层次聚类",
            "headline": "从最细的小簇一步步合并，重点看“合并顺序”和“树状图截断”。",
            "principle": "层次聚类会先把每个样本当作一个小簇，再按照距离不断合并，最终形成一棵从细到粗的层次树。",
            "visual_tip": "主图左侧看聚类散点，右侧看树状图，帮助学生把“合并历史”和“最终簇”对应起来。",
            "pros": "能保留层次结构，适合讲“先细后粗”的聚类思想。",
            "cons": "样本一多就会比较慢，而且不同 linkage 差异很大。",
            "fit": "适合样本量较小、层级结构明显的数据。",
        },
        "gmm": {
            "title": "高斯混合模型 GMM",
            "headline": "用多个高斯分量共同解释数据，重点看“椭圆分布”和“软分配概率”。",
            "principle": "GMM 假设数据由多个高斯分布混合而成，每个点并不是只属于一个簇，而是对每个簇都有一个概率归属。",
            "visual_tip": "主图会显示概率背景、椭圆边界和一个查询点的软分配概率。",
            "pros": "适合椭圆簇和重叠簇，能给出更细腻的概率解释。",
            "cons": "需要假设数据近似服从高斯混合，对参数初始化也比较敏感。",
            "fit": "适合椭圆簇、带重叠区域的数据。",
        },
    }
    return mapping[algorithm_key]


def dataset_overview(dataset_key):
    return {"title": get_dataset_label(dataset_key), "summary": get_dataset_summary(dataset_key)}


def parameter_explanation(algorithm_key, params, extras):
    if algorithm_key == "kmeans":
        return (
            "当前簇数 = `{0}`，初始化方式 = `{1}`，重复初始化次数 = `{2}`。"
            " 簇数决定最终会分成几组，初始化方式会影响中心从哪里出发。"
        ).format(params["n_clusters"], params["init"], params["n_init"])

    if algorithm_key == "dbscan":
        return (
            "当前 eps = `{0:.2f}`，min_samples = `{1}`。"
            " eps 决定邻域半径，min_samples 决定一个点要有多少邻居才算核心点。"
        ).format(params["eps"], params["min_samples"])

    if algorithm_key == "agg":
        return (
            "当前簇数 = `{0}`，linkage = `{1}`。"
            " linkage 决定两个簇之间的距离如何定义，所以会直接影响树状图的合并顺序。"
        ).format(params["n_clusters"], params["linkage"])

    return (
        "当前成分数 = `{0}`，协方差类型 = `{1}`，最大迭代次数 = `{2}`。"
        " 成分数决定模型想用多少个高斯分量解释数据，协方差类型决定椭圆有多灵活。"
    ).format(params["n_components"], params["covariance_type"], params["max_iter"])


def live_summary(algorithm_key, params, metrics, extras):
    if algorithm_key == "kmeans":
        if params["n_clusters"] < extras["suggested_clusters"]:
            return "当前簇数偏少，几个本来可以分开的簇被合并到了一起。"
        if params["n_clusters"] > extras["suggested_clusters"]:
            return "当前簇数偏多，原本完整的簇开始被切碎。"
        return "当前簇数和数据结构比较匹配，中心轨迹也能较清楚地显示收敛过程。"

    if algorithm_key == "dbscan":
        if metrics["noise_count"] > len(extras["labels"]) * 0.25:
            return "当前噪声点偏多，通常说明 eps 偏小或 min_samples 偏大。"
        if metrics["cluster_count"] <= 1:
            return "当前参数让多数样本被连成了一个大簇，说明密度判定偏宽。"
        return "当前 DBSCAN 能比较清楚地区分核心区域、边界区域和噪声点。"

    if algorithm_key == "agg":
        if params["linkage"] == "single":
            return "Single linkage 容易沿着近邻把簇串起来，桥接结构会特别明显。"
        if params["linkage"] == "complete":
            return "Complete linkage 更强调簇间最远距离，通常会让簇更紧凑。"
        return "当前 linkage 更像在平衡整体相似度，树状图的合并顺序会更平滑。"

    if extras["focus_entropy"] > 0.78:
        return "当前查询点在多个高斯分量之间摇摆，说明它处在重叠区域。"
    return "当前查询点对某一个高斯分量的归属概率更高，软分配结果比较明确。"


def bottom_conclusion(algorithm_key, dataset_key, metrics, extras):
    silhouette = metrics["silhouette"]
    if silhouette is None:
        score_text = "当前簇数不足或噪声过多，内部评价指标暂时没有稳定参考意义。"
    elif silhouette >= 0.45:
        score_text = "轮廓系数较高，说明簇内更紧、簇间更分开。"
    elif silhouette >= 0.20:
        score_text = "轮廓系数处于中等水平，当前分组有一定结构，但还不够明显。"
    else:
        score_text = "轮廓系数偏低，说明当前簇之间仍然混在一起。"

    extra_text = ""
    if algorithm_key == "dbscan":
        extra_text = " 当前噪声点数量为 `{0}`。".format(metrics["noise_count"])
    elif algorithm_key == "kmeans":
        extra_text = " 当前簇内平方误差约为 `{0:.2f}`。".format(extras["inertia"])
    elif algorithm_key == "gmm":
        extra_text = " 当前查询点的最高归属概率约为 `{0:.1f}%`。".format(extras["focus_max_probability"] * 100)

    return """
- 当前算法：`{0}`
- 当前数据集：`{1}`
- 当前簇数量：`{2}`
- 轮廓系数：`{3}`
- 教学结论：{4}{5}
""".format(
        ALGORITHM_LABELS[algorithm_key],
        get_dataset_label(dataset_key),
        metrics["cluster_count"],
        "--" if metrics["silhouette"] is None else "{0:.3f}".format(metrics["silhouette"]),
        score_text,
        extra_text,
    )
