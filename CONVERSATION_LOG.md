# AI vs Human 文章偵測器 - 開發對話記錄

**日期**：2025年12月10日  
**專案**：AIOT_HW5 - AI 偵測技術  
**最終部署狀態**：✅ 已成功部署至 Streamlit Cloud

---

## 📋 對話內容總結

### 第一階段：作業需求與規劃

**使用者需求**：
- 建立一個簡單的 AI vs Human 文章分類工具
- 使用者輸入文本 → 立即顯示判斷結果（AI% / Human%）
- 可採用 sklearn / transformers / 自建特徵法
- 使用 Streamlit 作為 UI
- 參考教學（test.html 的 AI 偵測技術高階理論篇）

**核心要求**：
- ✅ 最低需求：輸入文本 → 即時顯示 AI/Human 概率
- ✅ 可採用 sklearn 或自建特徵法
- ✅ Streamlit UI 框架
- ✅ 可視化統計量（可選）

---

### 第二階段：環境建置與套件安裝

**虛擬環境建立**：
```bash
cd c:\Users\Una\AIOT\AIOT_HW5
python -m venv venv
.\venv\Scripts\activate
```

**安裝所需套件**：
- streamlit >= 1.28.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- scikit-learn >= 0.24.0
- nltk >= 3.6.0

所有套件已成功安裝至虛擬環境。

---

### 第三階段：核心功能實作

**實現的特徵提取模組** (`AIDetectorFeatureExtractor`)：

#### 1. **句子節奏特徵 (Burstiness)**
- `sentence_length_mean`：平均句子長度
- `sentence_length_std`：句子長度標準差
- `burstiness`：句子節奏 (σ / μ)
  - AI 文本：低 burstiness（句子長度穩定）
  - Human 文本：高 burstiness（長短句交錯）

#### 2. **詞彙多樣性特徵**
- `type_token_ratio (TTR)`：不重複詞彙數 / 總詞彙數
- `lexical_diversity`：同 TTR
- `avg_word_length`：平均詞長
- `entropy_word_freq`：詞頻熵（衡量詞彙多樣性）

#### 3. **文風統計特徵 (Stylometry)**
- **Lexical 層**：
  - `function_word_ratio`：功能詞比例（的、了、和、是等）
  - `zipf_tail_ratio`：Zipf 長尾詞比例（罕見詞出現頻率）
  
- **Syntactic 層**：
  - `punctuation_ratio`：標點符號比例
  - `comma_ratio`：逗號比例
  - `common_connectors_ratio`：常見連接詞比例（因此、值得注意、總結等）

- **Emotion & Noise 層**：
  - `question_mark_ratio`：問號比例
  - `exclamation_ratio`：驚嘆號比例
  - `passive_voice_indicator`：被動語態指標

#### 4. **語意特徵**
- `repeated_structures`：重複句式比例
- `avg_entropy_per_sentence`：每句平均熵

**共 17 個精心設計的特徵**

---

### 第四階段：機器學習模型

**分類模型** (`AIDetectorModel`)：

1. **初版模型**：Logistic Regression
   - 使用合成訓練資料（50個AI樣本 + 50個Human樣本）
   - 訓練資料分布：
     - AI：較低 burstiness、連接詞多、Zipf長尾短
     - Human：較高 burstiness、詞彙多樣、長尾詞多

2. **改善版模型**（後來新增）：
   - 新增 RandomForestClassifier（200 棵樹）
   - **Ensemble 策略**：Logistic + RF 平均概率
   - **機率收縮**：向 0.5 靠攏（shrink_factor=0.8），避免過度自信的 100% 判定
   - **機率裁剪**：確保結果在 (0.01%, 99.99%) 之間

---

### 第五階段：NLP 工具鏈問題與解決

**問題 1：NLTK punkt_tab 缺失**
- **症狀**：`LookupError: punkt_tab not found`
- **原因**：NLTK 新版本需要 punkt_tab tokenizer
- **解決方案**：
  - 移除 NLTK 依賴
  - 自建簡易中文分詞函式：
    ```python
    def split_sentences(text: str):
        """以中文標點或換行切分句子"""
        parts = re.split(r'[。！？!?\n]+', text)
        sentences = [p.strip() for p in parts if p and p.strip()]
        return sentences
    
    def tokenize(text: str):
        """簡易詞元化：擷取連續中文字符或英數字序列"""
        tokens = re.findall(r'[\u4e00-\u9fff]+|[A-Za-z0-9]+', text)
        return tokens
    ```
  - 優點：避免外部依賴、支援中文、無額外下載

---

### 第六階段：概率過度自信問題與修正

**問題 2：AI 概率過高（100%）**
- **使用者反饋**：自己打的中文文本被判定為 99.6% AI，後來修正後仍為 99.99%
  
**使用者測試文本**（被過度判定為 AI）：
```
今天在IG上看到很多人post自己的Spotify年度回顧，我也去按了，沒想到我的最愛藝人是盧廣仲ㄟ！
真的是完全沒想過，因為他是我去完鹿港潮派才開始聽的，然後我朋友就傳了個那種可以在Spotify加入什麼派對的連結，
後來不知道為甚麼，他開的我沒辦法進去，就換我開他加進來，所以我是host所有的下一步都是我負責按，
沒想到我跟他的音樂品味相符度有85%，這也太高了吧，但不曉得他這個的判別標準是什麼
```

**修正策略**：
1. **特徵向量對齊**：確保合成訓練資料欄位順序與 `extractor.feature_names` 一致
2. **Ensemble 集成**：結合 LogisticRegression 和 RandomForest 的預測
3. **機率收縮**：
   ```python
   # 避免過度自信的 0% 或 100%
   shrink_factor = 0.8  # 向 0.5 靠攏
   ai_prob = 0.5 + (ai_prob - 0.5) * shrink_factor
   ai_prob = float(np.clip(ai_prob, 1e-4, 1 - 1e-4))
   ```

---

### 第七階段：Streamlit UI 設計

**UI 功能**：
1. **輸入區塊**：
   - 文本區域（至少 50 個字）
   - 「🔍 立即分析」按鈕

2. **結果展示**：
   - 🤖 AI 概率（百分比）
   - 👤 Human 概率（百分比）
   - 進度條視覺化（AI vs Human 比例）

3. **詳細特徵分析**（可選）：
   - **句子節奏特徵**：平均句長、句長標準差、Burstiness
   - **詞彙特徵**：TTR、平均詞長、詞頻熵
   - **結構特徵**：功能詞比例、連接詞比例、Zipf長尾比例
   - **標點特徵**：標點比例、逗號、問號、驚嘆號

4. **可視化圖表**（可選）：
   - 關鍵特徵值對比（水平條狀圖）
   - 句長分布直方圖

5. **側邊欄**：
   - 顯示詳細特徵（勾選框）
   - 顯示可視化圖表（勾選框）
   - 説明與使用說明

6. **判定結論**：
   - > 70%：🚨 很可能為 AI 生成
   - 50-70%：⚡ 混合特徵
   - < 50%：✅ 很可能為 Human 撰寫

---

### 第八階段：專案結構與檔案

**最終檔案清單**：
```
c:\Users\Una\AIOT\AIOT_HW5\
├── ai_detector.py              # 主應用程式（Streamlit）
├── requirements.txt            # 相依套件清單
├── README.md                   # 專案說明與使用方法
├── run.bat                     # Windows 快速啟動腳本
├── test.html                   # 教學資料（AI 偵測技術理論）
├── .gitignore                  # Git 忽略檔案清單
├── CONVERSATION_LOG.md         # 此文件（開發對話記錄）
└── venv\                       # Python 虛擬環境（已安裝所有套件）
```

---

### 第九階段：版本控制與部署

#### Git 初始化與推送
```bash
# 設定 Git 使用者
git config --global user.name "Una"
git config --global user.email "your-email@example.com"

# 初始化版本庫
git init
git add .
git commit -m "Initial commit: AI vs Human Article Detector with Streamlit UI"

# 設定遠端並推送
git remote add origin https://github.com/una0917/AIOT_HW5.git
git branch -M main
git push -u origin main
```

**GitHub 連結**：
- Repository：https://github.com/una0917/AIOT_HW5
- Branch：main
- Main file：ai_detector.py

#### Streamlit Cloud 部署
1. 訪問 https://streamlit.io/cloud
2. 登入（或用 GitHub 帳號註冊）
3. 點擊 **「New app」**
4. 選擇：
   - Repository：`una0917/AIOT_HW5`
   - Branch：`main`
   - Main file path：`ai_detector.py`
5. 點擊 **「Deploy」** 並等待 5-10 分鐘

**部署狀態**：✅ 已成功部署至 Streamlit Cloud

---

### 第十階段：快速開始指南

#### 本地運行
```bash
# 進入專案目錄
cd c:\Users\Una\AIOT\AIOT_HW5

# 啟動虛擬環境
.\venv\Scripts\Activate.ps1

# 安裝相依套件（如果尚未安裝）
pip install -r requirements.txt

# 啟動 Streamlit 應用
streamlit run ai_detector.py
```

應用將在 `http://localhost:8501` 打開。

#### 測試用範例文本
- **Human 風格**（應傾向 Human）：
  ```
  剛剛去市場買菜，遇到老朋友聊了很久，天氣特別好，心情也跟著放晴。晚餐想煮簡單的番茄炒蛋，家人說很期待。
  ```

- **AI 風格**（通常較規則）：
  ```
  本文將探討研究的背景、方法、結果與結論。首先介紹問題陳述，接著說明實驗設計，最後總結主要發現及未來方向。
  ```

---

## 🎯 技術亮點總結

### 1. **多維度特徵設計**
- 基於教學文檔的 17 個精心設計特徵
- 涵蓋：句子節奏、詞彙多樣性、文風統計、語意特徵等

### 2. **中文最佳化**
- 自建中文分詞與詞元化函式（無需 NLTK）
- 支援常見中文標點符號（。！？等）

### 3. **機器學習模型**
- Ensemble 策略（LogisticRegression + RandomForest）
- 機率校正（收縮與裁剪），避免過度自信

### 4. **完整 UI 體驗**
- Streamlit 現代化介面
- 詳細特徵分析與可視化
- 側邊欄配置與說明

### 5. **完善的版本控制與部署**
- GitHub 版本管理
- Streamlit Cloud 自動部署
- 可透過 git push 自動更新

---

## 📊 關鍵數據與設定

### 合成訓練資料分布
**AI 樣本特徵均值**：
- burstiness：0.25（低，句子穩定）
- common_connectors_ratio：0.5（高）
- zipf_tail_ratio：0.35（低，少用罕見詞）
- entropy_word_freq：3.5（相對低）

**Human 樣本特徵均值**：
- burstiness：0.65（高，句子波動大）
- common_connectors_ratio：0.15（低）
- zipf_tail_ratio：0.55（高，多用罕見詞）
- entropy_word_freq：4.5（相對高）

### 機率校正參數
- Ensemble：Logistic + RF 平均
- 收縮因子（shrink_factor）：0.8
- 機率範圍：[0.01%, 99.99%]

---

## ✅ 最終成果

### 完成清單
- ✅ 根據作業需求實現 AI vs Human 文章分類工具
- ✅ 特徵設計基於教學文檔的高階理論
- ✅ 使用 sklearn 機器學習框架
- ✅ Streamlit UI 實現即時判斷與可視化
- ✅ 修正概率過度自信問題
- ✅ 上傳 GitHub Repository
- ✅ 部署至 Streamlit Cloud（公開 demo）

### 可訪問資源
1. **本地應用**：
   ```bash
   streamlit run ai_detector.py
   ```

2. **GitHub Repository**：
   https://github.com/una0917/AIOT_HW5

3. **線上 Demo**（Streamlit Cloud）：
   部署完成後獲得的 streamlit.app 連結

4. **專案文檔**：
   - README.md：詳細使用說明與技術架構
   - CONVERSATION_LOG.md：此完整對話記錄

---

## 🔧 未來改進方向

1. **使用真實數據集訓練**
   - 收集真實 AI 與 Human 文本樣本
   - 改進模型準確率

2. **支援多語言**
   - 擴展中文到英文、日文等
   - 調整 tokenization 策略

3. **集成深度學習模型**
   - 使用 BERT / GPT embeddings
   - Hybrid 傳統特徵 + 深度語意

4. **XAI 可解釋性**
   - 新增 SHAP / LIME 視覺化
   - 展示模型關鍵決策因素

5. **API 服務**
   - Flask / FastAPI 後端
   - 供第三方應用呼叫

---

## 📝 重要提醒

⚠️ **免責聲明**：
- 該工具提供的是「訊號與機率」，而非「絕對判定」
- 不應作為學術倫理調查的唯一證據
- 對於邊界情況（40-60%）建議結合人工審查

---

**對話記錄建立日期**：2025年12月10日  
**最後更新**：部署至 Streamlit Cloud 完成  
**專案狀態**：🟢 已上線，可正常使用
