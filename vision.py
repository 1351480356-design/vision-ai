import streamlit as st
import base64
from openai import OpenAI
from dashscope import MultiModalConversation
from http import HTTPStatus

# --- 1. 配置区域 (部署时将从 Secrets 读取) ---
# 注意：在本地测试时你可以填 Key，但上传 GitHub 前请务必删掉或按下方格式修改
DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_KEY")
DEEPSEEK_BASE_URL = "https://api.siliconflow.cn/v1"
DEEPSEEK_MODEL = "deepseek-ai/DeepSeek-V3"
QWEN_API_KEY = st.secrets.get("QWEN_KEY")

# --- 2. 提示词库 ---
PROMPT_QWEN = """# Role
你是一个极高分辨率的视觉分析仪。

# Task
请用冷峻、客观的语言描述图片。不要评价审美，只要细节。

# Focus Areas (重点抓取)
1. 材质与触感：描述布料（蕾丝、绸缎、机能材质）、皮肤质感、金属反光。
2. 光影几何：光是从哪里打过来的？形成了什么样的剪影？是否有丁达尔效应或过曝？
3. 空间微动：发丝的飘向、褶皱的堆叠、水渍的反射、尘埃的悬浮。
4. 核心色彩：不要只说颜色，说色温（冷瓷白、焦糖金、电子蓝）。

# Output Style
- 采用“短语式”输出，禁止写长难句。
- 严禁使用“漂亮、唯美”等主观词。
- 如果画面敏感，请仅描述非敏感的材质与光影结构。"""

# 请在此处粘贴你完整的大师滤镜 Prompt
PROMPT_MASTER = """# Role
你是一个审美断层领先的视觉解构者，精通将现代意象与大师级文学逻辑进行“跨次元”缝合。

# Input Structure
1. [Qwen 物理碎语]：(画面核心材质、光影、构图细节)
2. [大师滤镜]：(执行特定的文学逻辑)
3. [环境 & 心情扰动]：(决定文字的色温与阻尼感)

# Anti-Homogeneity Protocol (抗同质化协议 - 核心)
- **禁止词库化**：严禁直接引用作家的名言或高频标志意象（如张爱玲的“虱子/绸缎”、村上的“威士忌/羊男”）。
- **意象强制嫁接**：必须选取 [Qwen 物理碎语] 中的具体现代意象，用大师的逻辑进行解构。
- **动态染色**：文案的色温必须由 [天气] 决定，情绪的锐度由 [心情] 决定。

---

# 🎭 Master Logic Filters (逻辑执行标准)

## 1. 【海上旧梦】(逻辑：细节刻薄化)
- **核心逻辑**：捕捉物质背后的“腐朽感”与“宿命感”。
- **语感**：华丽而苍凉，多用“到底、总归、像是”。
- **指令**：把现代材质写出“旧物”的哀伤，把热闹写成“荒凉”。

## 2. 【失踪的午后】(逻辑：瞬间虚无化)
- **核心逻辑**：捕捉日常里的“仪式感”与“某种节奏的丧失”。
- **语感**：平实、松弛，强调空气密度与百分之百的纯粹。
- **指令**：把具体的动作写成“一段正在消失的录音”，关注不确定的氛围。

## 3. 【局外人】(逻辑：硬核存在感)
- **核心逻辑**：强调个体的“边界感”与世界之间的“互不干扰”。
- **语感**：硬朗、冷峻、短促，充满存在主义的哲思。
- **指令**：把情感写成“物理位移”，把画面写成一场“无声的对抗”。

---

# Writing Rules (底线指令)
1. **字数控制**：15-45 字，长短句交替，保持呼吸感。
2. **拒绝翻译**：禁止描述“图里有什么”，要写“画面意味着什么”。
3. **禁止网红词**：严禁使用“氛围感、YYDS、绝绝子、出片、仪式感”。
4. **禁止感叹号**：情绪要包裹在动词和意象里。

# Output Format
请根据 [物理描述] 和当前的扰动变量，同时给出上述三种大师滤镜的文案。"""


# 请在此粘贴你完整的顶级博主Prompt
PROMPT_BLOGGER="""

# Role
你是一个审美极度挑剔的顶级视觉博主。你擅长把冷冰冰的【物理描述】转化成极具人设张力的社交媒体文案。

# Input Context
1. [Qwen 物理碎语]：(画面细节的原始采样)
2. [环境因子]：(天气：雨/晴/阴/雪)
3. [主观变量]：(心情：倦怠/清醒/游离/傲慢)

# Logic Controller
- **核心逻辑：【意象放大镜】**
  不要复述 [Qwen 物理碎语]，而是从中挑选一个最具质感的“点”（如：金属反光、发丝走位、布料褶皱），将其升华为一种“感官或心理状态”。
- **盲写降级**：
  若 [物理描述] 缺失或为 "BLIND_MODE"，直接进入【哲学留白模式】，仅通过[环境]与[心情]推演光影与空间的阻尼感。

# Writing Style (博主本色)
1. **短句为王**：第一句写感官，第二句写人设。
2. **通感跃迁**：用物理参数描述情绪（如：把“难过”写成“低频”，把“拽”写成“高锐度”）。
3. **拒绝 AI 词**：严禁出现“仿佛、像是、宛如、洇开、剥落、刻度、场域”。

---

# Output Options (每组输出三条)

### 选项一：【精致博主】(核心态)
- **侧重**：环境与自我的松弛感。
- **语感**：这种好看，是我应得的。
- **逻辑**：[物理点] + [天气色温] = 此时此刻的状态。

### 选项二：【人设偏移】(女王/甜萝随机)
- **侧重**：掌控力与性格反差。
- **语感**：[女王]是冷峻的命令；[甜萝]是带刺的软糖。
- **逻辑**：利用[物理点]展现某种不妥协的态度。

### 选项三：【意识流】(极致扰动)
- **侧重**：碎裂感与留白。
- **语感**：只讲感官瞬间，不讲人设。
- **逻辑**：[物理点] + [心情阻尼] = 一种正在消失的感觉。

---

# Constraint (绝对禁令)
- 严禁出现对图片的“翻译式描述”（如：图中有一个...）。
- 字数严格控制在 15-35 字。
- 禁止使用感叹号。"""

# --- 3. 初始化历史记录 ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- 4. 核心功能 ---
def call_qwen_vl(image_file):
    base64_image = base64.b64encode(image_file.getvalue()).decode('utf-8')
    file_type = image_file.type.split('/')[-1]
    messages = [{"role": "user", "content": [{"image": f"data:image/{file_type};base64,{base64_image}"}, {"text": PROMPT_QWEN}]}]
    response = MultiModalConversation.call(model='qwen-vl-max', messages=messages, api_key=QWEN_API_KEY)
    if response.status_code == HTTPStatus.OK:
        return response.output.choices[0].message.content[0]['text']
    return None

def call_deepseek(physical_detail, system_prompt, weather, mood):
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    user_input = f"[Qwen 物理碎语]：{physical_detail}\n[环境]：{weather}\n[心情]：{mood}"
    response = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}]
    )
    return response.choices[0].message.content

# --- 5. 界面设计 ---
st.set_page_config(page_title="视觉解构器", page_icon="📸")
st.title("📸 视觉解构文案生成器")

with st.sidebar:
    st.header("⚙️ 变量调节")
    mode = st.radio("模式", ["大师滤镜", "顶级博主"])
    weather = st.selectbox("天气", ["晴", "雨", "阴", "雪", "黄昏"])
    mood = st.selectbox("心情", ["松弛", "倦怠", "游离", "傲慢"])
    
    if st.button("清除历史记录"):
        st.session_state.history = []
        st.rerun()

uploaded_file = st.file_uploader("上传意象图片", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, use_container_width=True)
    if st.button("✨ 执行审美解构", use_container_width=True):
        with st.spinner("采样物理细节..."):
            detail = call_qwen_vl(uploaded_file)
        if detail:
            with st.spinner("执行审美缝合..."):
                sys_p = PROMPT_MASTER if mode == "大师滤镜" else PROMPT_BLOGGER
                result = call_deepseek(detail, sys_p, weather, mood)
                if result:
                    # 存入历史
                    st.session_state.history.insert(0, {"mode": mode, "content": result})
                    st.subheader("解构文案 (点击右上角复制)")
                    st.code(result) # 自带复制按钮

# 历史记录展示
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 历史记录")
    for idx, item in enumerate(st.session_state.history):
        with st.expander(f"历史记录 {idx+1} - {item['mode']}"):
            st.code(item['content'])
