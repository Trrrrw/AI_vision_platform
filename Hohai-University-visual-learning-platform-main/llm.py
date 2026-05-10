# pages/llm_chat.py
import streamlit as st
from openai import OpenAI

# ==================== 页面配置 ====================
# 注意：set_page_config 只能在每个页面的最顶部调用一次，且必须在其他st组件之前
# 由于主程序已经调用过，这里可以省略，或者用try包裹避免重复调用报错
try:
    st.set_page_config(page_title="国产大模型", page_icon="🤖", layout="wide")
except:
    pass

st.title("🤖 国产大模型体验区")
st.markdown("选择一个模型，输入问题，体验国产AI的能力。")

# ==================== 初始化会话状态 ====================
if "llm_messages" not in st.session_state:
    st.session_state.llm_messages = []

# ==================== 模型配置 ====================
MODEL_CONFIGS = {
    "DeepSeek": {
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "api_key": st.secrets.get("DEEPSEEK_API_KEY", "")
    },
    "智谱GLM": {
        "model": "glm-4-flash",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key": st.secrets.get("ZHIPU_API_KEY", "")
    },
    "通义千问": {
        "model": "qwen-turbo",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": st.secrets.get("ALIYUN_API_KEY", "")
    },
    "文心一言": {
        "model": "ernie-4.0-8k",
        "base_url": "https://qianfan.baidubce.com/v2",
        "api_key": st.secrets.get("BAIDU_API_KEY", "")
    }
}

# ==================== 侧边栏参数 ====================
with st.sidebar:
    st.header("⚙️ 模型设置")

    model_choice = st.selectbox("选择模型", list(MODEL_CONFIGS.keys()))

    temperature = st.slider(
        "Temperature", 0.0, 1.5, 0.7, 0.1,
        help="控制回复的随机性：值越高，回复越有创造性"
    )
    max_tokens = st.slider(
        "Max Tokens", 50, 2000, 1024, 50,
        help="控制回复的最大长度"
    )

    if st.button("🧹 清空对话历史", use_container_width=True):
        st.session_state.llm_messages = []
        st.rerun()

    st.divider()
    st.caption(f"当前模型: {model_choice}")

# ==================== 获取当前模型配置 ====================
config = MODEL_CONFIGS[model_choice]
if not config["api_key"]:
    st.error(
        f"❌ 未配置 {model_choice} 的 API Key。请在项目根目录的 `.streamlit/secrets.toml` 文件中添加 `{model_choice.upper()}_API_KEY`。")
    st.stop()

client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])

# ==================== 显示历史消息 ====================
for msg in st.session_state.llm_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==================== 处理用户输入 ====================
if prompt := st.chat_input(f"向 {model_choice} 提问..."):
    # 添加用户消息
    st.session_state.llm_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用模型
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            stream = client.chat.completions.create(
                model=config["model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.llm_messages
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.llm_messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"调用模型时出错: {e}")