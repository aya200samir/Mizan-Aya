import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import warnings
warnings.filterwarnings('ignore')

# ==================== مكتبات التعلم الآلي ====================
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# محاولة استيراد SHAP للتفسير
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ==================== إعدادات الصفحة ====================
st.set_page_config(
    page_title="Mizan AI - نظام العدالة الذكي",
    page_icon="⚖️",
    layout="wide"
)

# ==================== CSS مخصص ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;900&display=swap');
    * { font-family: 'Cairo', sans-serif; }
    
    .header {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        padding: 2rem;
        border-radius: 0 0 30px 30px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .header h1 { font-size: 3rem; font-weight: 900; margin-bottom: 0.5rem; }
    .header p { font-size: 1.2rem; opacity: 0.9; }
    
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #e0e0e0;
        height: 100%;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .metric-card .value { font-size: 2.2rem; font-weight: 900; color: #1e3c72; }
    .metric-card .label { color: #666; font-size: 1rem; }
    
    .bias-alert {
        background: #ffebee;
        border-right: 5px solid #f44336;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 600;
        color: #b71c1c;
    }
    
    .fairness-badge {
        background: #e8f5e9;
        border-right: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-weight: 600;
        color: #1b5e20;
    }
    
    .explanation-box {
        background: #f5f7fa;
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #ddd;
        margin: 1rem 0;
        direction: rtl;
        text-align: right;
    }
    
    .what-if-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #ced4da;
        margin: 1rem 0;
    }
    
    .footer {
        background: #1e3c72;
        color: white;
        padding: 1.5rem;
        border-radius: 30px 30px 0 0;
        margin-top: 3rem;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72, #2a5298);
        color: white;
        font-weight: 600;
        width: 100%;
        border: none;
        border-radius: 8px;
        padding: 0.75rem;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #2a5298, #1e3c72);
        box-shadow: 0 5px 15px rgba(30,60,114,0.4);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# ==================== توليد البيانات من طبقات متعددة ====================
def generate_official_data():
    """الطبقة الأولى: بيانات رسمية مجمعة (محاكاة لنشرة الإسكان)"""
    return pd.DataFrame({
        "المحافظة": ["القاهرة", "الجيزة", "الإسكندرية", "أسيوط", "سوهاج", "قنا", "أسوان", "المنيا"],
        "الوحدات": [15000, 12000, 10000, 8000, 7000, 6000, 5000, 5500],
        "نسبة_القبول": [0.18, 0.22, 0.20, 0.25, 0.27, 0.28, 0.30, 0.26]
    })

@st.cache_data
def generate_synthetic_data(n_extra=8000):
    """
    توليد بيانات صناعية محاكاة بناءً على البيانات الرسمية + خصائص إضافية
    """
    official = generate_official_data()
    official["المتقدمون_التقديريون"] = (official["الوحدات"] / official["نسبة_القبول"]).astype(int)
    total = n_extra

    # قائمة المحافظات الكاملة
    governorates_list = official["المحافظة"].tolist()
    
    # توزيع المحافظات حسب الوزن الرسمي
    probs = official["المتقدمون_التقديريون"] / official["المتقدمون_التقديريون"].sum()
    governorates = np.random.choice(governorates_list, size=total, p=probs)

    # المتغيرات الأساسية
    income = np.random.normal(5500, 2000, total).clip(1500, 15000)
    family_size = np.random.randint(1, 7, total)
    employment = np.random.choice(["رسمي", "غير رسمي"], total, p=[0.6, 0.4])
    
    # الحالة الاجتماعية
    marital_status = np.random.choice(
        ["أعزب", "متزوج", "مطلق", "أرمل"],
        total,
        p=[0.25, 0.60, 0.08, 0.07]
    )
    
    # الإعاقة
    disability = np.random.choice([0, 1], total, p=[0.885, 0.115])
    disability_severity = np.zeros(total)
    for i in range(total):
        if disability[i] == 1:
            disability_severity[i] = np.random.choice([0.3, 0.5, 0.8, 0.6, 0.5, 0.7, 0.9],
                                                       p=[0.25,0.15,0.1,0.12,0.1,0.15,0.13])
    
    # ملكية سابقة
    previous_ownership = np.random.choice([0, 1], total, p=[0.93, 0.07])

    data = pd.DataFrame({
        "المحافظة": governorates,
        "الدخل": income,
        "حجم_الأسرة": family_size,
        "نوع_العمل": employment,
        "الحالة_الاجتماعية": marital_status,
        "إعاقة": disability,
        "شدة_الإعاقة": disability_severity,
        "ملكية_سابقة": previous_ownership
    })

    # حساب الاستحقاق الفعلي
    ages = np.random.randint(18, 70, total)
    data["العمر"] = ages
    data["الاستحقاق_الفعلي"] = (
        (data["الدخل"] <= 6000) & 
        (data["ملكية_سابقة"] == 0) &
        (data["العمر"] >= 21)
    ).astype(int)

    # استثناءات إنسانية
    special_cases = (data["شدة_الإعاقة"] > 0.7) & (data["الدخل"] <= 7000)
    data.loc[special_cases, "الاستحقاق_الفعلي"] = 1

    # ===== نظام الأوزان التصاعدي المتراكم (Cumulative Progressive Weights) =====
    # الفلسفة: نبدأ من العجز البدني (الإعاقة الشديدة)، ثم الهشاشة الاجتماعية (الأرامل)، ثم المظلومية الجغرافية (المناطق النائية)
    
    data["وزن_العدالة"] = 1.0  # الوزن الأساسي

    # المرتبة الأولى: الإعاقة الشديدة (الوزن الأعلى - 2.0x)
    # الإعاقة الشديدة (أكثر من 0.7) تحصل على وزن مضاعف
    severe_disability_mask = data["شدة_الإعاقة"] >= 0.7
    data.loc[severe_disability_mask, "وزن_العدالة"] *= 2.0
    
    # الإعاقة المتوسطة (بين 0.4 و 0.7) تحصل على وزن 1.5x
    moderate_disability_mask = (data["شدة_الإعاقة"] >= 0.4) & (data["شدة_الإعاقة"] < 0.7)
    data.loc[moderate_disability_mask, "وزن_العدالة"] *= 1.5
    
    # الإعاقة البسيطة (أقل من 0.4) تحصل على وزن 1.2x
    mild_disability_mask = (data["شدة_الإعاقة"] > 0) & (data["شدة_الإعاقة"] < 0.4)
    data.loc[mild_disability_mask, "وزن_العدالة"] *= 1.2

    # المرتبة الثانية: الأرملة التي تعول (وزن - 1.8x)
    # الأرامل (خاصة مع وجود أطفال) يحصلن على وزن كبير
    widowed_mask = data["الحالة_الاجتماعية"] == "أرمل"
    data.loc[widowed_mask, "وزن_العدالة"] *= 1.8
    
    # المطلقات مع أطفال يحصلن على وزن 1.4x
    divorced_with_kids_mask = (data["الحالة_الاجتماعية"] == "مطلق") & (data["حجم_الأسرة"] > 2)
    data.loc[divorced_with_kids_mask, "وزن_العدالة"] *= 1.4

    # المرتبة الثالثة: المناطق النائية (وزن - 1.5x)
    remote_areas = ["أسيوط", "سوهاج", "قنا", "أسوان"]
    remote_mask = data["المحافظة"].isin(remote_areas)
    data.loc[remote_mask, "وزن_العدالة"] *= 1.5

    # العمالة غير المنتظمة (وزن إضافي 1.2x)
    informal_mask = data["نوع_العمل"] == "غير رسمي"
    data.loc[informal_mask, "وزن_العدالة"] *= 1.2

    # ===== الأوزان التقاطعية (Intersectionality) - تراكم الأوزان للحالات المركبة =====
    # مثال: أرملة + إعاقة + منطقة نائية = أوزان متراكمة
    # هذا يحدث تلقائياً لأننا نضرب الأوزان (الضرب المتتالي)
    # نحتاج فقط للتأكد من عدم تجاوز حد معين (لتجنب التضخم المفرط)
    data["وزن_العدالة"] = data["وزن_العدالة"].clip(upper=5.0)  # حد أقصى 5 أضعاف

    # القرار التقليدي
    data["القرار_التقليدي"] = (
        (data["الدخل"] <= 6000) & 
        (data["ملكية_سابقة"] == 0)
    ).astype(int)

    return data

# ==================== دالة MCAS لمكافحة الدقة الوهمية ====================
def mcas_score(y_true, y_pred, lambda1=1, lambda2=1):
    """
    حساب مقياس MCAS وفقًا لبحث د. محمد الهاداد
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    css_plus = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    css_minus = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0
    
    cfs = 0.5 * (
        (fp / (tp + tn + fp) if (tp + tn + fp) > 0 else 0) +
        (fn / (tp + tn + fn) if (tp + tn + fn) > 0 else 0)
    )
    
    mcas = (lambda1 * (css_plus - cfs) + lambda2 * (css_minus - cfs)) / (lambda1 + lambda2)
    return mcas

# ==================== تحليل EDA واكتشاف التحيز ====================
def analyze_bias(data):
    """تحليل معدلات القبول حسب الفئات المختلفة"""
    results = {}
    by_gov = data.groupby("المحافظة")["الاستحقاق_الفعلي"].mean()
    results["المحافظة"] = by_gov
    by_work = data.groupby("نوع_العمل")["الاستحقاق_الفعلي"].mean()
    results["نوع_العمل"] = by_work
    by_marital = data.groupby("الحالة_الاجتماعية")["الاستحقاق_الفعلي"].mean()
    results["الحالة_الاجتماعية"] = by_marital
    by_disability = data.groupby("إعاقة")["الاستحقاق_الفعلي"].mean()
    results["إعاقة"] = by_disability
    return results

def detect_bias_gap(data, feature):
    """حساب الفجوة بين أعلى وأقل نسبة قبول لميزة معينة"""
    rates = data.groupby(feature)["الاستحقاق_الفعلي"].mean()
    return rates.max() - rates.min()

# ==================== تدريب النموذج العادل ====================
@st.cache_resource
def train_fair_model(data):
    """
    تدريب RandomForest مع استخدام أوزان العينات (sample_weight) المستمدة من "وزن_العدالة"
    """
    feature_cols = ['العمر', 'الدخل', 'حجم_الأسرة', 'إعاقة', 'شدة_الإعاقة', 'ملكية_سابقة']
    
    data_encoded = data.copy()
    encoders = {}
    for col in ['المحافظة', 'نوع_العمل', 'الحالة_الاجتماعية']:
        le = LabelEncoder()
        data_encoded[col] = le.fit_transform(data_encoded[col])
        encoders[col] = le
        feature_cols.append(col)
    
    X = data_encoded[feature_cols]
    y = data_encoded['الاستحقاق_الفعلي']
    sample_weights = data_encoded['وزن_العدالة'].values

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42, stratify=y
    )

    fair_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    fair_model.fit(X_train, y_train, sample_weight=w_train)

    y_pred_fair = fair_model.predict(X_test)
    y_proba_fair = fair_model.predict_proba(X_test)[:, 1]

    metrics_fair = {
        'accuracy': accuracy_score(y_test, y_pred_fair),
        'precision': precision_score(y_test, y_pred_fair),
        'recall': recall_score(y_test, y_pred_fair),
        'f1': f1_score(y_test, y_pred_fair),
        'mcas': mcas_score(y_test, y_pred_fair)
    }

    return {
        'model': fair_model,
        'encoders': encoders,
        'feature_cols': feature_cols,
        'metrics': metrics_fair,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred_fair,
        'y_proba': y_proba_fair,
        'sample_weights': w_test,
        'X_train': X_train,
        'y_train': y_train
    }

# ==================== تدريب النموذج التقليدي ====================
def train_traditional_model(data):
    """نموذج تقليدي بسيط: لا يستخدم أوزان عدالة"""
    feature_cols = ['العمر', 'الدخل', 'حجم_الأسرة', 'إعاقة', 'شدة_الإعاقة', 'ملكية_سابقة']
    data_encoded = data.copy()
    encoders = {}
    for col in ['المحافظة', 'نوع_العمل', 'الحالة_الاجتماعية']:
        le = LabelEncoder()
        data_encoded[col] = le.fit_transform(data_encoded[col])
        encoders[col] = le
        feature_cols.append(col)
    
    X = data_encoded[feature_cols]
    y = data_encoded['الاستحقاق_الفعلي']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    trad_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    trad_model.fit(X_train, y_train)
    y_pred = trad_model.predict(X_test)
    y_proba = trad_model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'mcas': mcas_score(y_test, y_pred)
    }
    
    return {
        'model': trad_model,
        'encoders': encoders,
        'feature_cols': feature_cols,
        'metrics': metrics,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'X_train': X_train,
        'y_train': y_train
    }

# ==================== دالة التنبؤ + النظام الهجين ====================
def hybrid_decision(model_pack, user_data, threshold_high=0.8, threshold_low=0.2):
    """
    - إذا كانت الثقة ≥ 0.8 → مقبول آلياً مع تفسير
    - إذا كانت الثقة ≤ 0.2 → مرفوض آلياً مع تفسير
    - وإلا → يحتاج مراجعة بشرية مع تقرير
    """
    model = model_pack['model']
    encoders = model_pack['encoders']
    feature_cols = model_pack['feature_cols']
    
    input_df = pd.DataFrame([user_data])
    for col, encoder in encoders.items():
        if col in input_df.columns:
            input_df[col] = encoder.transform(input_df[col])
    
    X_input = input_df[feature_cols]
    prob = model.predict_proba(X_input)[0][1]
    pred = model.predict(X_input)[0]
    
    if prob >= threshold_high:
        decision = "✅ مقبول تلقائياً"
        confidence = prob
        review_needed = False
    elif prob <= threshold_low:
        decision = "❌ مرفوض تلقائياً"
        confidence = 1 - prob
        review_needed = False
    else:
        decision = "⚠️ يحتاج مراجعة بشرية"
        confidence = prob
        review_needed = True
    
    return {
        'prediction': pred,
        'probability': prob,
        'decision': decision,
        'confidence': confidence,
        'review_needed': review_needed
    }

def generate_explanation(user_data, hybrid_result):
    """توليد تقرير تفسيري بالعربية"""
    factors = []
    
    # الشروط الأساسية
    if user_data['الدخل'] <= 6000:
        factors.append("✓ الدخل مناسب (≤ 6000)")
    else:
        factors.append("✗ الدخل مرتفع")
    
    if user_data['ملكية_سابقة'] == 0:
        factors.append("✓ لا توجد ملكية سابقة")
    else:
        factors.append("✗ لديه ملكية سابقة")
    
    if user_data['العمر'] >= 21:
        factors.append("✓ العمر مناسب")
    else:
        factors.append("✗ العمر أقل من 21")
    
    # الأوزان التصاعدية
    if user_data['إعاقة'] == 1:
        severity = user_data['شدة_الإعاقة']
        if severity >= 0.7:
            factors.append(f"✓✓ إعاقة شديدة (درجة {severity:.1f}) - أولوية قصوى")
        elif severity >= 0.4:
            factors.append(f"✓ إعاقة متوسطة (درجة {severity:.1f}) - أولوية عالية")
        else:
            factors.append(f"✓ إعاقة بسيطة (درجة {severity:.1f}) - أولوية")
    
    if user_data['الحالة_الاجتماعية'] == 'أرمل':
        factors.append("✓✓ أرمل/أرملة - أولوية اجتماعية قصوى")
    elif user_data['الحالة_الاجتماعية'] == 'مطلق' and user_data.get('حجم_الأسرة', 1) > 2:
        factors.append("✓ مطلق/مطلقة مع أطفال - أولوية اجتماعية")
    
    remote_areas = ["أسيوط", "سوهاج", "قنا", "أسوان"]
    if user_data['المحافظة'] in remote_areas:
        factors.append("✓ من منطقة نائية - أولوية جغرافية")
    
    if user_data['نوع_العمل'] == 'غير رسمي':
        factors.append("✓ عمالة غير منتظمة - أولوية اقتصادية")
    
    explanation = f"""
    ### 📋 تقرير تفسير القرار
    **النتيجة:** {hybrid_result['decision']}  
    **الثقة:** {hybrid_result['confidence']*100:.1f}%  
    
    **العوامل المؤثرة:**
    """ + "\n".join([f"- {f}" for f in factors])
    
    if hybrid_result['review_needed']:
        explanation += "\n\n**🔔 تم تحويل الطلب للمراجعة البشرية لعدم وضوح الحالة.**"
    else:
        explanation += "\n\n**🤖 تم اتخاذ القرار آلياً بناءً على وضوح الحالة.**"
    
    return explanation

# ==================== تحليل "ماذا لو" (What-if Analysis) ====================
def what_if_analysis(model_pack, base_user_data):
    """
    تحليل تأثير تغيير الدخل على فرص الاستحقاق
    """
    model = model_pack['model']
    encoders = model_pack['encoders']
    feature_cols = model_pack['feature_cols']
    
    income_range = np.arange(2000, 10001, 500)
    probabilities = []
    
    for inc in income_range:
        temp_data = base_user_data.copy()
        temp_data['الدخل'] = inc
        
        input_df = pd.DataFrame([temp_data])
        for col, encoder in encoders.items():
            if col in input_df.columns:
                input_df[col] = encoder.transform(input_df[col])
        
        X_input = input_df[feature_cols]
        prob = model.predict_proba(X_input)[0][1]
        probabilities.append(prob)
    
    return income_range, probabilities

# ==================== الصفحة الرئيسية للتطبيق ====================
def main():
    st.markdown("""
    <div class="header">
        <h1>⚖️ Mizan AI - نظام العدالة الذكي للإسكان الاجتماعي</h1>
        <p>نموذج هجين مع نظام الأوزان التصاعدي المتراكم (الإعاقة ← الأرامل ← المناطق النائية)</p>
    </div>
    """, unsafe_allow_html=True)

    # ===== توليد البيانات =====
    with st.spinner("📊 جاري توليد بيانات المحاكاة..."):
        data = generate_synthetic_data(n_extra=8000)
        official = generate_official_data()

    # ===== تحليل EDA والتحيز =====
    st.markdown("## 📊 تحليل البيانات الاستكشافي (EDA) واكتشاف التحيز")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("إجمالي العينات", f"{len(data):,}")
    with col2:
        st.metric("نسبة المستحقين الفعلية", f"{data['الاستحقاق_الفعلي'].mean()*100:.1f}%")
    with col3:
        st.metric("نسبة القرار التقليدي", f"{data['القرار_التقليدي'].mean()*100:.1f}%")
    
    bias_gap = detect_bias_gap(data, "المحافظة")
    st.metric("الفجوة بين المحافظات (تحيز)", f"{bias_gap*100:.1f}%")
    
    st.markdown("### 📈 توزيع الاستحقاق حسب الفئات")
    tab1, tab2, tab3, tab4 = st.tabs(["المحافظة", "نوع العمل", "الحالة الاجتماعية", "الإعاقة"])
    
    bias_results = analyze_bias(data)
    with tab1:
        fig = px.bar(x=bias_results["المحافظة"].index, y=bias_results["المحافظة"].values,
                     title="نسبة الاستحقاق حسب المحافظة", color=bias_results["المحافظة"].values,
                     color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)
        if bias_gap > 0.1:
            st.markdown(f'<div class="bias-alert">⚠️ تحذير: فجوة كبيرة بين المحافظات ({bias_gap*100:.1f}%)</div>', unsafe_allow_html=True)
    
    with tab2:
        fig = px.bar(x=bias_results["نوع_العمل"].index, y=bias_results["نوع_العمل"].values,
                     title="نسبة الاستحقاق حسب نوع العمل", color=bias_results["نوع_العمل"].values,
                     color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        fig = px.bar(x=bias_results["الحالة_الاجتماعية"].index, y=bias_results["الحالة_الاجتماعية"].values,
                     title="نسبة الاستحقاق حسب الحالة الاجتماعية", color=bias_results["الحالة_الاجتماعية"].values,
                     color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)
    with tab4:
        fig = px.bar(x=['غير معاق', 'معاق'], y=bias_results["إعاقة"].values,
                     title="نسبة الاستحقاق حسب الإعاقة", color=bias_results["إعاقة"].values,
                     color_continuous_scale="RdYlGn")
        st.plotly_chart(fig, use_container_width=True)

    # عرض توزيع أوزان العدالة
    st.markdown("### ⚖️ توزيع أوزان العدالة")
    fig = px.histogram(data, x="وزن_العدالة", nbins=50, title="توزيع أوزان العدالة",
                       color_discrete_sequence=["#4caf50"])
    fig.add_vline(x=1.0, line_dash="dash", line_color="red", annotation_text="الوزن الأساسي")
    st.plotly_chart(fig, use_container_width=True)

    # ===== تدريب النماذج (تقليدي وعادل) =====
    st.markdown("---")
    st.markdown("## 🤖 تدريب النماذج والمقارنة")

    with st.spinner("🔄 جاري تدريب النموذج التقليدي..."):
        trad_pack = train_traditional_model(data)
    with st.spinner("⚖️ جاري تدريب النموذج العادل (مع أوزان العدالة التصاعدية)..."):
        fair_pack = train_fair_model(data)

    # عرض مقارنة الأداء
    st.markdown("### 📊 مقارنة أداء النموذجين")
    comp_df = pd.DataFrame({
        'المقياس': ['الدقة (Accuracy)', 'الدقة (Precision)', 'الاستدعاء (Recall)', 'F1', 'MCAS'],
        'النموذج التقليدي': [
            f"{trad_pack['metrics']['accuracy']*100:.2f}%",
            f"{trad_pack['metrics']['precision']*100:.2f}%",
            f"{trad_pack['metrics']['recall']*100:.2f}%",
            f"{trad_pack['metrics']['f1']*100:.2f}%",
            f"{trad_pack['metrics']['mcas']*100:.2f}%"
        ],
        'النموذج العادل': [
            f"{fair_pack['metrics']['accuracy']*100:.2f}%",
            f"{fair_pack['metrics']['precision']*100:.2f}%",
            f"{fair_pack['metrics']['recall']*100:.2f}%",
            f"{fair_pack['metrics']['f1']*100:.2f}%",
            f"{fair_pack['metrics']['mcas']*100:.2f}%"
        ]
    })
    st.dataframe(comp_df, use_container_width=True)

    # مصفوفات الارتباك
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**مصفوفة ارتباك - النموذج التقليدي**")
        cm_trad = confusion_matrix(trad_pack['y_test'], trad_pack['y_pred'])
        fig = px.imshow(cm_trad, text_auto=True, x=['غير مستحق', 'مستحق'], y=['غير مستحق', 'مستحق'],
                        color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**مصفوفة ارتباك - النموذج العادل**")
        cm_fair = confusion_matrix(fair_pack['y_test'], fair_pack['y_pred'])
        fig = px.imshow(cm_fair, text_auto=True, x=['غير مستحق', 'مستحق'], y=['غير مستحق', 'مستحق'],
                        color_continuous_scale='Greens')
        st.plotly_chart(fig, use_container_width=True)

    # تحليل التحيز بعد التدريب
    st.markdown("### ⚖️ تحليل العدالة بعد تطبيق النموذج العادل")
    test_data = data.iloc[fair_pack['X_test'].index].copy()
    test_data['تنبؤ_عادل'] = fair_pack['y_pred']
    
    acc_by_gov_fair = test_data.groupby('المحافظة')['تنبؤ_عادل'].mean()
    acc_by_gov_true = test_data.groupby('المحافظة')['الاستحقاق_الفعلي'].mean()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=acc_by_gov_fair.index, y=acc_by_gov_fair.values, name='النموذج العادل', marker_color='#4caf50'))
    fig.add_trace(go.Bar(x=acc_by_gov_true.index, y=acc_by_gov_true.values, name='الاستحقاق الفعلي', marker_color='#2196f3'))
    fig.update_layout(title='مقارنة القبول حسب المحافظة: النموذج العادل vs الاستحقاق الفعلي',
                      xaxis_title='المحافظة', yaxis_title='نسبة القبول', barmode='group')
    st.plotly_chart(fig, use_container_width=True)

    new_gap = acc_by_gov_fair.max() - acc_by_gov_fair.min()
    st.metric("الفجوة الجديدة بين المحافظات (بعد النموذج العادل)", f"{new_gap*100:.1f}%",
              delta=f"{(bias_gap - new_gap)*100:.1f}% انخفاض", delta_color="normal")

    # ===== النظام الهجين =====
    st.markdown("---")
    st.markdown("## 🧠 النظام الهجين للقرارات")
    st.info("""
    **آلية العمل:**
    - **المنطقة الخضراء (ثقة ≥ 80%)** → قرار آلي (مقبول) مع تفسير.
    - **المنطقة الحمراء (ثقة ≤ 20%)** → قرار آلي (مرفوض) مع تفسير.
    - **المنطقة الرمادية (بين 20% و 80%)** → تحويل للمراجعة البشرية مع تقرير تفسيري مفصل.
    """)

    # إدخال بيانات المتقدم
    with st.expander("➕ أدخل بيانات المتقدم", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("العمر", 18, 70, 35)
            governorate = st.selectbox("المحافظة", ['القاهرة', 'الجيزة', 'الإسكندرية', 'أسيوط', 'سوهاج', 'قنا', 'أسوان', 'المنيا'])
            employment = st.selectbox("نوع العمل", ['رسمي', 'غير رسمي'])
            marital = st.selectbox("الحالة الاجتماعية", ['أعزب', 'متزوج', 'مطلق', 'أرمل'])
        with col2:
            income = st.number_input("الدخل الشهري", 1500, 15000, 5000)
            family_size = st.number_input("حجم الأسرة", 1, 6, 3)
            disability = st.checkbox("لديه إعاقة")
            disability_severity = st.slider("شدة الإعاقة (إذا وجدت)", 0.0, 1.0, 0.5, step=0.1,
                                            disabled=not disability)
            previous = st.checkbox("ملكية سابقة")

    col1, col2 = st.columns(2)
    with col1:
        predict_button = st.button("🔮 تنبؤ وتحليل", use_container_width=True)
    with col2:
        what_if_button = st.button("📊 تحليل ماذا لو (What-if)", use_container_width=True)

    if predict_button or what_if_button:
        # تجهيز بيانات المستخدم
        user_data = {
            'العمر': age,
            'الدخل': income,
            'حجم_الأسرة': family_size,
            'إعاقة': 1 if disability else 0,
            'شدة_الإعاقة': disability_severity if disability else 0,
            'ملكية_سابقة': 1 if previous else 0,
            'المحافظة': governorate,
            'نوع_العمل': employment,
            'الحالة_الاجتماعية': marital
        }

        if predict_button:
            # التنبؤ بالنموذج العادل
            result = hybrid_decision(fair_pack, user_data)
            explanation = generate_explanation(user_data, result)

            # عرض النتيجة
            if "مقبول" in result['decision']:
                st.success(f"### {result['decision']}")
            elif "مرفوض" in result['decision']:
                st.error(f"### {result['decision']}")
            else:
                st.warning(f"### {result['decision']}")

            st.progress(result['probability'])
            st.markdown(f"**الثقة:** {result['confidence']*100:.1f}%")
            st.markdown(explanation, unsafe_allow_html=True)

            if result['review_needed']:
                st.markdown("""
                <div style="background:#fff3cd; padding:1rem; border-radius:10px; border-right:5px solid #ff9800;">
                    <strong>📢 توصية:</strong> يُرجى عرض الطلب على اللجنة المختصة مع التقرير أعلاه.
                </div>
                """, unsafe_allow_html=True)

        if what_if_button:
            st.markdown("### 📈 تحليل ماذا لو (What-if)")
            income_range, probs = what_if_analysis(fair_pack, user_data)
            
            fig = px.line(x=income_range, y=probs, markers=True,
                         title="تأثير تغيير الدخل على فرص الاستحقاق",
                         labels={'x': 'الدخل الشهري', 'y': 'احتمالية الاستحقاق'})
            fig.add_hline(y=0.8, line_dash="dash", line_color="green", annotation_text="حد القبول الآلي")
            fig.add_hline(y=0.2, line_dash="dash", line_color="red", annotation_text="حد الرفض الآلي")
            fig.update_layout(yaxis_range=[0,1])
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            <div class="what-if-card">
                <strong>🔍 تحليل:</strong><br>
                - عند الدخل الحالي ({income} جنيه)، فرصتك: {probs[np.abs(income_range - income).argmin()]*100:.1f}%<br>
                - الدخل المطلوب لتحقيق 80% فرصة: {income_range[np.where(probs >= 0.8)[0][0]] if any(p >= 0.8 for p in probs) else 'لا يمكن'} جنيه<br>
                - الدخل الذي يخفض الفرصة لأقل من 20%: {income_range[np.where(probs <= 0.2)[0][-1]] if any(p <= 0.2 for p in probs) else 'لا يمكن'} جنيه
            </div>
            """, unsafe_allow_html=True)

    # ===== تقييم النظام =====
    st.markdown("---")
    st.markdown("## 📝 تقييم النظام وفلسفة العدالة")
    
    # حساب متوسط الأوزان للفئات المختلفة
    remote_areas = ["أسيوط", "سوهاج", "قنا", "أسوان"]
    avg_weight_remote = data[data["المحافظة"].isin(remote_areas)]["وزن_العدالة"].mean()
    avg_weight_widowed = data[data["الحالة_الاجتماعية"] == "أرمل"]["وزن_العدالة"].mean()
    avg_weight_severe_disability = data[data["شدة_الإعاقة"] >= 0.7]["وزن_العدالة"].mean()
    avg_weight_intersectional = data[(data["الحالة_الاجتماعية"] == "أرمل") & 
                                      (data["إعاقة"] == 1) & 
                                      (data["المحافظة"].isin(remote_areas))]["وزن_العدالة"].mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background:white; padding:2rem; border-radius:15px; box-shadow:0 5px 20px rgba(0,0,0,0.05);">
            <h4>✨ نقاط القوة في هذا التصميم:</h4>
            <ul>
                <li><strong>نظام الأوزان التصاعدي:</strong> يبدأ من العجز البدني (الإعاقة الشديدة)، ثم الهشاشة الاجتماعية (الأرامل)، ثم المظلومية الجغرافية (المناطق النائية).</li>
                <li><strong>الأوزان التقاطعية:</strong> تراكم الأوزان للحالات المركبة (أرملة + إعاقة + منطقة نائية) يعطي وزناً مضاعفاً يعكس واقع الحياة.</li>
                <li><strong>مقياس MCAS:</strong> مكافحة الدقة الوهمية والتركيز على أداء النموذج مع الفئات القليلة.</li>
                <li><strong>نظام هجين:</strong> الجمع بين السرعة (حالات واضحة) والدقة البشرية (حالات حدية).</li>
                <li><strong>تحليل ماذا لو:</strong> شفافية كاملة تسمح للمسؤول بتجربة سيناريوهات مختلفة.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background:white; padding:2rem; border-radius:15px; box-shadow:0 5px 20px rgba(0,0,0,0.05);">
            <h4>⚖️ تأثير الأوزان التصاعدية:</h4>
            <ul>
                <li><strong>الإعاقة الشديدة:</strong> متوسط الوزن {avg_weight_severe_disability:.2f}x</li>
                <li><strong>الأرامل:</strong> متوسط الوزن {avg_weight_widowed:.2f}x</li>
                <li><strong>المناطق النائية:</strong> متوسط الوزن {avg_weight_remote:.2f}x</li>
                <li><strong>التقاطع (أرملة + إعاقة + نائية):</strong> متوسط الوزن {avg_weight_intersectional:.2f}x</li>
            </ul>
            <p>النموذج العادل يحقق توازناً بين الكفاءة والعدالة، ويقلل الفجوات بين الفئات.</p>
        </div>
        """, unsafe_allow_html=True)

    # تذييل
    st.markdown("""
    <div class="footer">
        <p>⚖️ Mizan AI - نظام العدالة الذكي | مستند إلى أبحاث د. محمد الهاداد (MCAS) | © 2026</p>
        <p>فلسفة التصميم: الإعاقة ← الأرامل ← المناطق النائية ← أوزان متراكمة للتداخلات</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
