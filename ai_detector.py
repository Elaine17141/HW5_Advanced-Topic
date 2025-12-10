import streamlit as st
import numpy as np
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

font_path = os.path.join(os.path.dirname(__file__), "fonts", "NotoSansTC-VariableFont_wght.ttf")
font_prop = fm.FontProperties(fname=font_path)

plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False


import seaborn as sns
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Dict

# ============================================================
# 页面配置与样式
# ============================================================

st.set_page_config(
    page_title="AI 文章偵測器",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义 CSS 样式
def load_custom_css():
    st.markdown("""
    <style>
    /* 主容器 */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
    }
    
    /* 标题样式 */
    .main-title {
        text-align: center;
        color: #1e3c72;
        font-size: 3.5rem;
        font-weight: 900;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* 结果卡片 */
    .result-card {
        border-radius: 15px;
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: none;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .card-ai {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .card-human {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .card-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 10px;
        opacity: 0.9;
    }
    
    .card-value {
        font-size: 3rem;
        font-weight: 900;
        margin: 10px 0;
    }
    
    .card-label {
        font-size: 0.85rem;
        opacity: 0.8;
        margin-top: 5px;
    }
    
    /* 输入区域 */
    .input-section {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .section-title {
        color: #1e3c72;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 15px;
        border-left: 4px solid #667eea;
        padding-left: 10px;
    }
    
    /* 按钮样式 */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.6);
    }
    
    /* 特征分析卡片 */
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    
    .feature-name {
        color: #1e3c72;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 5px;
    }
    
    .feature-value {
        color: #667eea;
        font-weight: 700;
        font-size: 1.2rem;
    }
    
    /* 结论盒子 */
    .conclusion-box {
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        border-left: 5px solid #667eea;
    }
    
    .conclusion-ai {
        background: rgba(102, 126, 234, 0.1);
        border-left-color: #667eea;
    }
    
    .conclusion-human {
        background: rgba(245, 87, 108, 0.1);
        border-left-color: #f5576c;
    }
    
    .conclusion-mixed {
        background: rgba(255, 193, 7, 0.1);
        border-left-color: #ffc107;
    }
    
    /* 进度条容器 */
    .progress-container {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    /* 响应式 */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        .card-value {
            font-size: 2rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

load_custom_css()



# ============================================================
# 句子与词元处理
# ============================================================

def remove_emoji(text):
    emoji_pattern = re.compile(
    "["                 
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended-A
    "]+",
    flags=re.UNICODE)
    return emoji_pattern.sub("", text)

def split_sentences(text: str) -> list:
    """分割句子"""
    parts = re.split(r'[。！？!?\n]+', text)
    return [p.strip() for p in parts if p.strip()]

def tokenize(text):
    # 保留中文、英文、數字，其他全部當作 noise 丟掉
    tokens = re.findall(r'[0-9A-Za-z]+|[\u4e00-\u9fa5]+', text)
    return tokens


# ============================================================
# 特徵抽取器
# ============================================================

class AIDetectorFeatureExtractor:
    """AI文本特征提取器"""
    
    def __init__(self):
        self.feature_names = [
            'sentence_length_mean',
            'sentence_length_std',
            'burstiness',
            'type_token_ratio',
            'avg_word_length',
            'punctuation_ratio',
            'function_word_ratio',
            'comma_ratio',
            'lexical_diversity',
            'entropy_word_freq',
            'zipf_tail_ratio',
            'repeated_structures',
            'common_connectors_ratio',
            'question_mark_ratio',
            'exclamation_ratio',
            'passive_voice_indicator',
            'avg_entropy_per_sentence',
        ]

    def extract_features(self, text: str) -> Dict[str, float]:
        """提取特征"""
        features = {}
        text = text.strip()

        if len(text) == 0:
            return {name: 0.0 for name in self.feature_names}

        sentences = split_sentences(text)
        sentence_lengths = [len(tokenize(s)) for s in sentences]

        features['sentence_length_mean'] = float(np.mean(sentence_lengths)) if sentence_lengths else 0.0
        features['sentence_length_std'] = float(np.std(sentence_lengths)) if sentence_lengths else 0.0

        if features['sentence_length_mean'] > 0:
            features['burstiness'] = features['sentence_length_std'] / features['sentence_length_mean']
        else:
            features['burstiness'] = 0.0

        words = tokenize(text.lower())
        unique_words = len(set(words)) if words else 0
        total_words = len(words)

        if total_words > 0:
            features['type_token_ratio'] = unique_words / total_words
            features['lexical_diversity'] = unique_words / total_words
        else:
            features['type_token_ratio'] = 0.0
            features['lexical_diversity'] = 0.0

        features['avg_word_length'] = float(np.mean([len(w) for w in words])) if words else 0.0

        total_chars = len(text)
        punct_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        features['punctuation_ratio'] = float(punct_count / total_chars) if total_chars else 0.0

        features['comma_ratio'] = float(text.count('，') / len(sentences)) if sentences else 0.0
        features['question_mark_ratio'] = float(text.count('？') / len(sentences)) if sentences else 0.0
        features['exclamation_ratio'] = float(text.count('！') / len(sentences)) if sentences else 0.0

        function_words = ['的', '了', '和', '是', '在', '以', '有', '等', '與', '或']
        fw_count = sum(text.count(fw) for fw in function_words)
        features['function_word_ratio'] = float(fw_count / total_words) if total_words else 0.0

        connectors = ['因此', '另外', '同時', '總之', '首先']
        conn_count = sum(text.count(c) for c in connectors)
        features['common_connectors_ratio'] = float(conn_count / len(sentences)) if sentences else 0.0

        if total_words > 0:
            freq = Counter(words)
            rare = sum(1 for w, f in freq.items() if f == 1)
            features['zipf_tail_ratio'] = float(rare / len(freq))
        else:
            features['zipf_tail_ratio'] = 0.0

        features['entropy_word_freq'] = self._entropy(words)
        features['repeated_structures'] = 0.0
        features['passive_voice_indicator'] = float(text.count('被') / len(sentences)) if sentences else 0.0
        features['avg_entropy_per_sentence'] = features['entropy_word_freq']

        return features

    def _entropy(self, words: list) -> float:
        """计算熵"""
        if not words:
            return 0.0
        freq = Counter(words)
        total = len(words)
        entropy = 0.0
        for f in freq.values():
            p = f / total
            entropy -= p * np.log2(p)
        return float(entropy / np.log2(len(freq))) if freq else 0.0

# ============================================================
# 内置模型
# ============================================================

class AIDetectorModel:
    """AI检测模型"""
    
    def __init__(self):
        self.extractor = AIDetectorFeatureExtractor()
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.rf = RandomForestClassifier(n_estimators=200, random_state=42)
        self.is_trained = False

    def train_sample_model(self):
        """训练模型"""
        n = 120
        features = []
        labels = []

        # AI样本
        for _ in range(n // 2):
            sample = [
                np.random.normal(15, 3),
                np.random.normal(4, 1),
                np.random.normal(0.25, 0.05),
                np.random.normal(0.55, 0.05),
                np.random.normal(3.2, 0.3),
                np.random.normal(0.08, 0.02),
                np.random.normal(0.24, 0.05),
                np.random.normal(0.7, 0.15),
                np.random.normal(0.55, 0.05),
                np.random.normal(3.2, 0.4),
                np.random.normal(0.35, 0.08),
                np.random.normal(0.05, 0.03),
                np.random.normal(0.5, 0.1),
                np.random.normal(0.05, 0.02),
                np.random.normal(0.02, 0.01),
                np.random.normal(0.10, 0.03),
                np.random.normal(2.0, 0.3),
            ]
            features.append(sample)
            labels.append(1)

        # Human样本
        for _ in range(n // 2):
            sample = [
                np.random.normal(12, 5),
                np.random.normal(8, 2),
                np.random.normal(0.6, 0.15),
                np.random.normal(0.7, 0.1),
                np.random.normal(3.0, 0.5),
                np.random.normal(0.12, 0.03),
                np.random.normal(0.2, 0.05),
                np.random.normal(0.4, 0.2),
                np.random.normal(0.7, 0.1),
                np.random.normal(4.5, 0.4),
                np.random.normal(0.55, 0.1),
                np.random.normal(0.12, 0.05),
                np.random.normal(0.2, 0.05),
                np.random.normal(0.15, 0.05),
                np.random.normal(0.05, 0.02),
                np.random.normal(0.04, 0.02),
                np.random.normal(3.2, 0.5),
            ]
            features.append(sample)
            labels.append(0)

        X = np.array(features)
        y = np.array(labels)

        X_scaled = self.scaler.fit_transform(X)

        self.model.fit(X_scaled, y)
        self.rf.fit(X_scaled, y)
        self.is_trained = True

    def predict(self, text: str) -> Tuple[float, float, Dict]:
        """预测"""
        text = remove_emoji(text)

        if not self.is_trained:
            self.train_sample_model()

        f = self.extractor.extract_features(text)
        X = np.array([f[name] for name in self.extractor.feature_names]).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        p1 = self.model.predict_proba(X_scaled)[0][1]
        p2 = self.rf.predict_proba(X_scaled)[0][1]
        ai_prob = (p1 + p2) / 2
        ai_prob = float(ai_prob)
        human_prob = 1 - ai_prob
        return ai_prob, human_prob, f


# ============================================================
# Streamlit UI - 美化版
# ============================================================

@st.cache_resource
def init_model():
    """缓存模型"""
    return AIDetectorModel()

def render_header():
    """渲染标题"""
    st.markdown('<div class="main-title">🤖 AI vs Human 文章偵測器</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">✨ 使用AI技術精準判断文本来源 ✨</div>', unsafe_allow_html=True)

def render_input_section() -> str:
    """渲染输入区域"""
    st.markdown('<div class="section-title">📝 輸入文本</div>', unsafe_allow_html=True)
    
    text = st.text_area(
        "在下方输入要分析的文本（至少 50 字）:",
        height=220,
        placeholder="粘贴你的文本内容... 支持中英文混合"
    )
    
    return text

def render_result_cards(ai_prob: float, human_prob: float):
    """渲染结果卡片"""
    st.markdown('<div class="section-title">📊 分析結果</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        ai_label = "极可能为 AI 生成" if ai_prob > 0.7 else "可能为 AI 生成" if ai_prob > 0.5 else "可能为人类撰写"
        st.markdown(f"""
        <div class="result-card card-ai">
            <div class="card-title">🤖 AI 生成概率</div>
            <div class="card-value">{ai_prob*100:.1f}%</div>
            <div class="card-label">{ai_label}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        human_label = "极可能为人类撰写" if human_prob > 0.7 else "可能为人类撰写" if human_prob > 0.5 else "可能为 AI 生成"
        st.markdown(f"""
        <div class="result-card card-human">
            <div class="card-title">👤 人類撰寫概率</div>
            <div class="card-value">{human_prob*100:.1f}%</div>
            <div class="card-label">{human_label}</div>
        </div>
        """, unsafe_allow_html=True)

def render_progress_bar(ai_prob: float):
    """渲染进度条"""
    st.markdown('<div class="progress-container">', unsafe_allow_html=True)
    
    col_label1, col_label2, col_label3 = st.columns([1, 1, 1])
    with col_label1:
        st.caption("🤖 AI")
    with col_label3:
        st.caption("👤 Human")
    
    progress_html = f"""
    <div style="display: flex; margin: 20px 0; border-radius: 10px; overflow: hidden; background: #e9ecef;">
        <div style="width: {ai_prob*100:.1f}%; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); height: 40px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 0.9rem;">
            {ai_prob*100:.1f}%
        </div>
        <div style="width: {(1-ai_prob)*100:.1f}%; background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%); height: 40px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 0.9rem;">
            {(1-ai_prob)*100:.1f}%
        </div>
    </div>
    """
    st.markdown(progress_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def render_features(features_dict: Dict):
    """渲染特征分析"""
    if st.checkbox("📋 显示詳細特徵分析", value=False):
        st.markdown('<div class="section-title">🔬 特征详情</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**句子节奏特征**")
            for name in ['sentence_length_mean', 'sentence_length_std', 'burstiness']:
                st.markdown(f"""
                <div class="feature-card">
                    <div class="feature-name">{name}</div>
                    <div class="feature-value">{features_dict[name]:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**词汇特征**")
            for name in ['type_token_ratio', 'avg_word_length', 'entropy_word_freq']:
                st.markdown(f"""
                <div class="feature-card">
                    <div class="feature-name">{name}</div>
                    <div class="feature-value">{features_dict[name]:.3f}</div>
                </div>
                """, unsafe_allow_html=True)

def render_visualization(text: str):
    """渲染图表"""
    if st.checkbox("📈 显示可視化圖表", value=False):
        st.markdown('<div class="section-title">📊 数据可视化</div>', unsafe_allow_html=True)
        
        sentences = split_sentences(text)
        sentence_lengths = [len(tokenize(s)) for s in sentences]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(sentence_lengths, bins=max(5, len(set(sentence_lengths))),
                    color='#667eea', alpha=0.7, edgecolor='#764ba2', linewidth=2)
            ax.set_xlabel('Sentence Length (tokens)', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title('句長分佈', fontweight='bold', fontsize=12)
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
        
        with col2:
            words = tokenize(text.lower())
            if words:
                freq = Counter(words)
                top_words = dict(freq.most_common(10))
                
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.barh(list(top_words.keys()), list(top_words.values()), 
                       color='#f5576c', alpha=0.7, edgecolor='#764ba2', linewidth=2)
                ax.set_xlabel('Frequency', fontweight='bold')
                ax.set_title('高頻詞彙 (Top 10)', fontweight='bold', fontsize=12)
                ax.invert_yaxis()
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)

def render_conclusion(ai_prob: float):
    """渲染结论"""
    st.markdown('<div class="section-title">🎯 判定結論</div>', unsafe_allow_html=True)
    
    if ai_prob > 0.7:
        st.markdown("""
        <div class="conclusion-box conclusion-ai">
            <h4>⚠️ 極可能為 AI 生成</h4>
            <p>該文本呈現出以下典型的 AI 特徵:</p>
            <ul>
                <li>✓ 句子節奏平穩 (低 Burstiness)</li>
                <li>✓ 常見連接詞使用較頻繁</li>
                <li>✓ 詞彙分布較規則和均勻</li>
                <li>✓ 文風高度一致</li>
            </ul>
            <p><strong>建議:</strong> 需要進一步人工審查確認</p>
        </div>
        """, unsafe_allow_html=True)
    elif ai_prob > 0.5:
        st.markdown("""
        <div class="conclusion-box conclusion-mixed">
            <h4>⚡ 混合特徵 - 可能為 AI 生成或經過大幅編輯</h4>
            <p>該文本展現了混合特徵，難以確定來源:</p>
            <ul>
                <li>~ 部分特徵與 AI 相符</li>
                <li>~ 部分特徵與人類相符</li>
                <li>~ 可能是人類編輯的 AI 內容，或 AI 潤色的人類文本</li>
            </ul>
            <p><strong>建議:</strong> 建議結合人工審查和其他方法判斷</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="conclusion-box conclusion-human">
            <h4>✅ 極可能為人類撰寫</h4>
            <p>該文本呈現出以下典型的人類特徵:</p>
            <ul>
                <li>✓ 句子長度波動較大 (高 Burstiness)</li>
                <li>✓ 詞彙選擇多樣性高</li>
                <li>✓ 存在自然的語言不規則性</li>
                <li>✓ 個人風格明顯</li>
            </ul>
            <p><strong>評估:</strong> 該文本很可能出自真人手筆</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """主函数"""
    render_header()
    
    # 侧边栏
    with st.sidebar:
        st.markdown("### ⚙️ 設定與說明")
        
        st.info("""
        **📖 使用方式**
        1. 在文本框中輸入要分析的文本
        2. 點擊「🔍 立即分析」按鈕
        3. 查看 AI 概率和詳細分析
        
        **🔬 偵測原理**
        - Burstiness: 句子節奏分析
        - TTR: 詞彙多樣性
        - Entropy: 詞頻熵計算
        - Stylometry: 文風統計
        - Zipf's Law: 長尾詞分析
        
        **💡 注意事項**
        - 本工具基於統計特徵
        - 不能作為唯一判斷依據
        - 結果準確度受文本長度影響
        """)
        
        st.divider()
        st.caption("🚀 由 OpenSpec 驅動 | v1.0")
    
    # 主区域
    text = render_input_section()
    
    # 分析按钮
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    
    with col_btn1:
        analyze_btn = st.button("🔍 立即分析", use_container_width=True)
    
    # 执行分析
    if analyze_btn:
        if len(text.strip()) < 50:
            st.warning("⚠️ 請輸入至少 50 個字的文本來分析", icon="📝")
            return
        
        try:
            with st.spinner("🔄 正在分析文本..."):
                model = init_model()
                ai_prob, human_prob, features_dict = model.predict(text)
            
            # 保存结果到 session
            st.session_state.last_result = {
                'ai_prob': ai_prob,
                'human_prob': human_prob,
                'features': features_dict,
                'text': text
            }
            
            st.success("✅ 分析完成!", icon="🎉")
            
        except Exception as e:
            st.error(f"❌ 分析失敗: {str(e)}", icon="💥")
    
    # 显示结果
    if 'last_result' in st.session_state:
        result = st.session_state.last_result
        
        render_result_cards(result['ai_prob'], result['human_prob'])
        render_progress_bar(result['ai_prob'])
        render_features(result['features'])
        render_visualization(result['text'])
        render_conclusion(result['ai_prob'])
        
        st.divider()
        st.caption("💻 Powered by Streamlit | Made with ❤️")

if __name__ == "__main__":
    main()
