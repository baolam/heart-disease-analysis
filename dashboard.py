import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from collections import defaultdict

rf_imp  = [0.00378944, 0.06451446, 0.44498126, 0.24994588, 0.12873835, 0.09088146, 0.01522668, 0.00192247]
xgb_imp = [0.03188968, 0.11535992, 0.2500725,  0.23829542, 0.13508256, 0.11364076, 0.06017521, 0.05548387]

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CVD Risk Factor Analysis",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

    .stApp { background-color: #0f1117; color: #e8eaf0; }

    .main-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #0f1117 100%);
        border-left: 4px solid #e05c5c;
        padding: 24px 32px; margin-bottom: 32px;
        border-radius: 0 8px 8px 0;
    }
    .main-header h1 {
        font-family: 'IBM Plex Mono', monospace; color: #e8eaf0;
        font-size: 1.8rem; margin: 0 0 6px 0; letter-spacing: -0.5px;
    }
    .main-header p { color: #8b90a0; margin: 0; font-size: 0.85rem; font-weight: 300; }
    .main-header span { color: #e05c5c; font-weight: 600; }

    .metric-card {
        background: #1a1f2e; border: 1px solid #2a2f3e;
        border-radius: 8px; padding: 20px 24px;
        text-align: center; transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #e05c5c; }
    .metric-label {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
        color: #8b90a0; text-transform: uppercase;
        letter-spacing: 1px; margin-bottom: 8px;
    }
    .metric-value {
        font-family: 'IBM Plex Mono', monospace; font-size: 2rem;
        font-weight: 600; color: #e8eaf0; line-height: 1;
    }
    .metric-sub { font-size: 0.75rem; color: #e05c5c; margin-top: 6px; font-weight: 600; }

    .section-title {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
        color: #e05c5c; text-transform: uppercase; letter-spacing: 2px;
        margin: 32px 0 16px 0; display: flex; align-items: center; gap: 12px;
    }
    .section-title::after { content: ''; flex: 1; height: 1px; background: #2a2f3e; }

    .chart-container {
        background: #1a1f2e; border: 1px solid #2a2f3e;
        border-radius: 8px; padding: 20px;
    }
    .insight-box {
        background: #1a1f2e; border-left: 3px solid #e05c5c;
        border-radius: 0 6px 6px 0; padding: 14px 18px;
        margin: 8px 0; font-size: 0.85rem; color: #c8cad4;
    }
    .insight-box strong { color: #e8eaf0; }

    .sql-block {
        background: #0d1117; border: 1px solid #2a2f3e;
        border-radius: 6px; padding: 16px;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
        color: #79c0ff; overflow-x: auto; margin: 12px 0;
    }
    .badge {
        display: inline-block; padding: 2px 10px; border-radius: 20px;
        font-size: 0.7rem; font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
    }
    .badge-best { background: #1a3a2a; color: #4ade80; border: 1px solid #4ade80; }
    .badge-good { background: #1a2a3a; color: #60a5fa; border: 1px solid #60a5fa; }
    .badge-ok   { background: #2a2a1a; color: #f59e0b; border: 1px solid #f59e0b; }
    .badge-base { background: #2a1a1a; color: #f87171; border: 1px solid #f87171; }

    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MATPLOTLIB STYLE — khớp distribution_plots
# ─────────────────────────────────────────────
CHART_STYLE = {
    'figure.facecolor': '#F8F9FA',
    'axes.facecolor':   '#FFFFFF',
    'axes.edgecolor':   '#DEE2E6',
    'axes.labelcolor':  '#495057',
    'xtick.color':      '#6C757D',
    'ytick.color':      '#6C757D',
    'grid.color':       '#E9ECEF',
    'font.family':      'DejaVu Sans',
}

ACCENT  = '#E63946'
ACCENT2 = '#4361EE'
ACCENT3 = '#2DC653'
PALETTE = ['#4361EE','#3A86FF','#4CC9F0','#4895EF',
           '#560BAD','#7209B7','#F72585','#B5179E']

LABELS_VI = [
    'Air pollution', 'Alcohol consumption', 'BMI', 'Cholesterol',
    'Diabetes', 'Glucose', 'Physical activities', 'Tobacco'
]

# ─────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    conn = sqlite3.connect("sample_v4.db")
    df_raw = pd.read_sql("SELECT * FROM NearsestSample", conn)
    conn.close()

    df = df_raw[(df_raw["TimeDim"] >= 2010) & (df_raw["TimeDim"] <= 2015)].reset_index(drop=True)
    df = df.drop_duplicates()
    df = df.drop(columns=["id", "x7", "SpatialDim"], errors="ignore")
    df.columns = [
        'cardiovascular_diseases', 'air_pollution', 'alcohol_consumption',
        'BMI', 'cholesterol', 'diabetes', 'glucose',
        'physical_activities', 'tobacco', 'time'
    ]
    df = df.fillna(df.mean(numeric_only=True))
    df['air_pollution'] = np.log1p(df['air_pollution'])

    cols = df.columns[:-1]
    mask = pd.Series(True, index=df.index)
    for col in cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        mask &= (df[col] >= Q1 - 3*IQR) & (df[col] <= Q3 + 3*IQR)
    df = df[mask]
    return df

@st.cache_data
def train_all_models(_df):
    feature_names = ['air_pollution','alcohol_consumption','BMI','cholesterol',
                     'diabetes','glucose','physical_activities','tobacco']
    X = _df[feature_names]
    print(X.describe())
    y = _df['cardiovascular_diseases']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree':     DecisionTreeRegressor(max_depth=5, min_samples_split=10,
                                                   min_samples_leaf=5, random_state=42),
        'Extra Trees':       ExtraTreesRegressor(n_estimators=200, max_depth=5, random_state=42),
        'Random Forest':     RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42),
        'XGBoost':           XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.8,
                                          random_state=42, verbosity=0),
    }
    results = {}
    for name, m in models.items():
        m.fit(Xtr, y_train)
        yp = m.predict(Xte)
        imp = (np.abs(m.coef_)/np.abs(m.coef_).sum()
               if hasattr(m, 'coef_') else m.feature_importances_)
        results[name] = {
            'rmse': mean_squared_error(y_test, yp) ** 0.5,
            'r2':  r2_score(y_test, yp),
            'imp': imp,
            'y_pred': yp,
        }
    results['_y_test'] = y_test.values
    results['_features'] = feature_names
    return results, models, scaler

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
try:
    df = load_and_preprocess()
    # results, models, scaler = train_all_models(df)
except Exception as e:
    st.error(f"⚠️ Unable to load database: {e}")
    st.stop()

feature_names = ['air_pollution','alcohol_consumption','BMI','cholesterol',
                 'diabetes','glucose','physical_activities','tobacco']

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;
                color:#e05c5c;letter-spacing:2px;margin-bottom:20px;'>
        ● CONTROL PANEL
    </div>""", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📊 Overview", "🔍 EDA", "📈 Visualization", "🤖 Modeling", "🗄️ SQL Analysis", "🏗️ Project Architecture"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
    <div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:#8b90a0;'>
        Dataset: WHO GHO API<br>
        Period: 2010–2015<br>
        Records (clean): 10,516<br>
        Countries: 166<br>
        Features: 8<br>
        Target: CVD Rate
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🫀 CVD Risk Factor Analysis</h1>
    <p>Cardiovascular Disease Prediction · <span>WHO GHO Dataset</span> ·
       2010–2015 · Machine Learning Analysis</p>
</div>""", unsafe_allow_html=True)

# =================================================================
# PAGE: OVERVIEW
# =================================================================
if page == "📊 Overview":

    cols = st.columns(5)
    kpis = [
        ("TOTAL RECORDS", "10,956", "before cleaning"),
        ("CLEAN RECORDS", "8,605", "after IQR×3 filter"),
        ("COUNTRIES",     "166",    "spatial coverage"),
        ("FEATURES",      "8",      "risk factors used"),
        ("BEST R²",       "0.6020", "XGBoost"),
    ]
    for col, (label, value, sub) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)
    
        # --- PHẦN BỔ SUNG: XU HƯỚNG VÀ TỶ LỆ CÁC YẾU TỐ ---
    st.markdown('<div class="section-title">Risk Factors Distribution & Global Trends</div>', unsafe_allow_html=True)
    
    col_pie, col_line = st.columns(2)

    groups_def = {
        'Lifestyle\n(BMI, Tobacco, Alcohol consumption, Physical activities)':
            ['physical_activities','tobacco','alcohol_consumption','BMI'],
        'Environment\n(Air pollution)': ['air_pollution'],
        'Clinial factors\n(Cholesterol, Diabetes, Glucose)':
            ['cholesterol','diabetes','glucose'],
    }
    feat_map = {f: i for i, f in enumerate(feature_names)}
    group_vals = {}
    for name, comps in groups_def.items():
        group_vals[name] = sum(xgb_imp[feat_map[c]] for c in comps)

    # col1, col2 = st.columns([1, 2])
    with col_pie:
        # fig, ax = plt.subplots(figsize=(5, 5))
        # wedges, _, autotexts = ax.pie(
        #     group_vals.values(), labels=None,
        #     autopct='%1.2f%%', colors=['#4361EE','#F72585','#7209B7'],
        #     startangle=90, pctdistance=0.68,
        #     wedgeprops=dict(linewidth=2.5, edgecolor='#F8F9FA')
        # )
        # for at in autotexts:
        #     at.set_fontsize(11); at.set_fontweight('bold'); at.set_color('white')
        # ax.legend(list(group_vals.keys()), loc='lower center',
        #           bbox_to_anchor=(0.5, -0.3), ncol=1, fontsize=9,
        #           framealpha=0, labelcolor='#212529')
        # ax.set_title('Contribution by Group \n(XGBoost)', fontsize=10.5,
        #              fontweight='bold', color='#212529', pad=14)
        # fig.tight_layout()
        # # st.pyplot(fig)
        # st.plotly_chart(fig)
        # plt.close()
        # 1. Chuẩn bị dữ liệu từ dictionary group_vals
        labels = list(group_vals.keys())
        values = list(group_vals.values())

        # 2. Vẽ biểu đồ bằng Plotly
        fig = px.pie(
            names=labels, 
            values=values,
            title='Contribution by Group<br>(XGBoost)',
            color_discrete_sequence=['#4361EE', '#F72585', '#7209B7'] # Giữ đúng bộ màu của chủ nhân
        )

        # 3. Tinh chỉnh hiển thị (giống các thiết lập wedges và autotexts)
        fig.update_traces(
            textinfo='percent', # Chỉ hiện phần trăm (giống autopct='%1.2f%%')
            textfont_size=14,
            textfont_color='white',
            marker=dict(line=dict(color='#F8F9FA', width=2.5)), # Giống wedgeprops
            hovertemplate="<b>%{label}</b><br>Tỷ lệ: %{percent:.2%}<br>Giá trị: %{value}<extra></extra>"
        )

        # 4. Thiết lập bố cục (Layout)
        fig.update_layout(
            title_font=dict(size=14, color='#212529', family="Arial Black"),
            legend=dict(
                orientation="h",       # Chú thích nằm ngang
                yanchor="bottom",
                y=-0.2,                # Đưa chú thích xuống dưới giống bbox_to_anchor
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=80, b=50, l=20, r=20),
            width=500,
            height=550
        )

        # 5. Hiển thị lên Streamlit
        st.plotly_chart(fig, use_container_width=True)

    # with col_pie:
    #     # 1. Biểu đồ tròn: Tỷ lệ đóng góp của các yếu tố nguy cơ (Feature Importance)
    #     # Dựa trên dữ liệu chủ nhân cung cấp: BMI (36.71%) + Cholesterol (35.61%) + Others
    #     pie_data = pd.DataFrame({
    #         'Factor': ['BMI', 'Cholesterol', 'Blood Pressure', 'Age', 'Smoking', 'Others'],
    #         'Importance': [36.71, 35.61, 12.5, 8.2, 4.5, 2.48]
    #     })
        
    #     fig_pie = px.pie(pie_data, values='Importance', names='Factor', 
    #                      title='<b>Risk Factor Importance (XGBoost)</b>',
    #                      hole=0.4, # Tạo hình vòng nhẫn (Donut chart) cho hiện đại
    #                      color_discrete_sequence=px.colors.qualitative.Pastel)
        
    #     fig_pie.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=350)
    #     st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<div class="section-title">Relationship between features and target</div>', unsafe_allow_html=True)
    features_all = feature_names + ['cardiovascular_diseases']
    corr_matrix = df[features_all].corr() # Lưu ý: Tương quan không đổi khi dùng StandardScaler nên có thể dùng trực tiếp df

    # 2. Chuẩn bị dữ liệu cho Plotly
    z = corr_matrix.values
    x = list(corr_matrix.columns)
    y = list(corr_matrix.index)

    # 3. Tạo Heatmap bằng Figure Factory (để dễ dàng hiển thị con số 'annot')
    fig = ff.create_annotated_heatmap(
        z=z,
        x=x,
        y=y,
        annotation_text=np.around(z, decimals=2), # Làm tròn 2 chữ số (fmt='.2f')
        colorscale='RdBu', # Bảng màu tương tự 'RdBu_r'
        zmin=-1, zmax=1,
        showscale=True
    )

    # 4. Tinh chỉnh giao diện cho "sang chảnh"
    fig.update_layout(
        title='Correlation Matrix — Risk Factors and CVD Rate',
        title_x=0.5, # Căn giữa tiêu đề
        title_font=dict(size=14, color='#212529', family="Arial Black"),
        margin=dict(t=100, l=150), # Chừa khoảng trống cho nhãn trục Y
        xaxis=dict(side='bottom'), # Đưa nhãn trục X xuống dưới
        width=800,
        height=700
    )

    # Tinh chỉnh font chữ cho các con số bên trong ô
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 8

    # 5. Hiển thị lên Streamlit
    st.plotly_chart(fig, use_container_width=True)

    with col_line:
        # 2. Biểu đồ đường: Xu hướng tỷ lệ CVD qua các năm (2010 - 2015)
        trend_data = pd.DataFrame({
            'Year': [2010, 2011, 2012, 2013, 2014, 2015],
            'CVD Rate': [39.55, 39.68, 39.82, 39.95, 40.05, 40.13]
        })
        
        fig_line = px.line(trend_data, x='Year', y='CVD Rate',
                           title='<b>Global CVD Rate Trend (2010-2015)</b>',
                           markers=True, 
                           line_shape='spline') # Đường cong mềm mại
        
        # Trang trí đường biểu đồ cho giống tone màu chủ đạo
        fig_line.update_traces(line_color='#e05c5c', line_width=3)
        fig_line.update_layout(
            xaxis_title="Year",
            yaxis_title="Rate per 100k",
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=50, b=20),
            height=350
        )
        # Thêm lưới mờ cho trục Y
        fig_line.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        st.plotly_chart(fig_line, use_container_width=True)
    
    # st.markdown('<div class="section-title">SQL Queries & Result</div>', unsafe_allow_html=True)

    try:
        conn_sql = sqlite3.connect("sample_v4.db")
    except Exception as e:
        st.error(f"Can not connect to DB: {e}"); st.stop()

    queries_display = [
        {
            "title": "Annual CVD Trends (2010–2015)",
            "question": "How has the CVD rate changed over the years?",
            "sql": """SELECT TimeDim AS year, COUNT(*) AS total_records,
            ROUND(AVG(y),2) AS avg_cvd, ROUND(MIN(y),2) AS min_cvd, ROUND(MAX(y),2) AS max_cvd
            FROM NearsestSample
            WHERE TimeDim BETWEEN 2010 AND 2015
            GROUP BY TimeDim ORDER BY TimeDim""",
        },
        {
            "title": "Top 10 Highest CVD Prevalence Countries",
            "question": "Which country carries the greatest CVD burden?",
            "sql": """SELECT SpatialDim AS country, ROUND(AVG(y),2) AS avg_cvd, COUNT(*) AS records
            FROM NearsestSample
            WHERE TimeDim BETWEEN 2010 AND 2015 AND y IS NOT NULL
            GROUP BY SpatialDim ORDER BY avg_cvd DESC LIMIT 10""",
        },
        {
            "title": "Average Risk Factors by Year",
            "question": "How do metabolic factors change over time?",
            "sql": """SELECT TimeDim AS year,
            ROUND(AVG(x3),2) AS avg_BMI, ROUND(AVG(x4),2) AS avg_cholesterol,
            ROUND(AVG(x5),2) AS avg_diabetes, ROUND(AVG(x6),2) AS avg_glucose,
            ROUND(AVG(x8),2) AS avg_physical_activities, ROUND(AVG(x9),2) AS avg_tobacco
            FROM NearsestSample
            WHERE TimeDim BETWEEN 2010 AND 2015
            GROUP BY TimeDim ORDER BY TimeDim""",
        },
        {
            "title": "Analysis of Missing Data",
            "question": "What is the missing rate for each feature, and how does it justify the infrastructure type?",
            "sql": """SELECT 'infrastructure(dropped)' AS feature,
        COUNT(*)-COUNT(x7) AS missing,
        ROUND(100.0*(COUNT(*)-COUNT(x7))/COUNT(*),2) AS pct_missing
    FROM NearsestSample WHERE TimeDim BETWEEN 2010 AND 2015
    UNION ALL SELECT 'air_pollution', COUNT(*)-COUNT(x1),
        ROUND(100.0*(COUNT(*)-COUNT(x1))/COUNT(*),2)
    FROM NearsestSample WHERE TimeDim BETWEEN 2010 AND 2015
    UNION ALL SELECT 'alcohol_consumption', COUNT(*)-COUNT(x2),
        ROUND(100.0*(COUNT(*)-COUNT(x2))/COUNT(*),2)
    FROM NearsestSample WHERE TimeDim BETWEEN 2010 AND 2015
    ORDER BY pct_missing DESC""",
        },
        {
            "title": "Descriptive Statistics for Variables",
            "question": "What are the descriptive statistics for each risk factor?",
            "sql": """SELECT 'cardiovascular_diseases' AS variable,
            ROUND(AVG(y),2) AS mean, ROUND(MIN(y),2) AS min, ROUND(MAX(y),2) AS max
        FROM NearsestSample WHERE TimeDim BETWEEN 2010 AND 2015
        UNION ALL SELECT 'BMI', ROUND(AVG(x3),2), ROUND(MIN(x3),2), ROUND(MAX(x3),2)
        FROM NearsestSample WHERE TimeDim BETWEEN 2010 AND 2015
        UNION ALL SELECT 'cholesterol', ROUND(AVG(x4),2), ROUND(MIN(x4),2), ROUND(MAX(x4),2)
        FROM NearsestSample WHERE TimeDim BETWEEN 2010 AND 2015
        UNION ALL SELECT 'diabetes', ROUND(AVG(x5),2), ROUND(MIN(x5),2), ROUND(MAX(x5),2)
        FROM NearsestSample WHERE TimeDim BETWEEN 2010 AND 2015
        UNION ALL SELECT 'tobacco', ROUND(AVG(x9),2), ROUND(MIN(x9),2), ROUND(MAX(x9),2)
        FROM NearsestSample WHERE TimeDim BETWEEN 2010 AND 2015""",
        }
    ]

    for q in queries_display:
        with st.expander(f"**{q['title']}**", expanded=True):
            st.markdown(
                f"<div style='color:#8b90a0;font-size:0.82rem;margin-bottom:10px;'>"
                f"❓ {q['question']}</div>", unsafe_allow_html=True)
            # st.markdown(f"<div class='sql-block'>{q['sql']}</div>", unsafe_allow_html=True)
            try:
                result_df = pd.read_sql(q['sql'], conn_sql)
                st.dataframe(result_df, use_container_width=True)
            except Exception as e:
                st.error(f"Query error: {e}")

    conn_sql.close()

    st.markdown('<div class="section-title">Model Performance — All models</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <div class="chart-container">
        <table style="width:100%;border-collapse:collapse;
                      font-family:IBM Plex Mono,monospace;font-size:0.82rem;">
            <thead>
                <tr style="border-bottom:2px solid #e05c5c;">
                    <th style="text-align:left;padding:8px;color:#8b90a0;">MODEL</th>
                    <th style="text-align:right;padding:8px;color:#8b90a0;">RMSE</th>
                    <th style="text-align:right;padding:8px;color:#8b90a0;">R²</th>
                    <th style="text-align:center;padding:8px;color:#8b90a0;">STATUS</th>
                </tr>
            </thead>
            <tbody>
                <tr style="border-bottom:1px solid #2a2f3e;">
                    <td style="padding:9px 8px;color:#e8eaf0;">Linear Regression</td>
                    <td style="text-align:right;padding:9px 8px;color:#f87171;">12.53</td>
                    <td style="text-align:right;padding:9px 8px;color:#f87171;">0.0956</td>
                    <td style="text-align:center;padding:9px 8px;">
                        <span class="badge badge-base">Baseline</span></td>
                </tr>
                <tr style="border-bottom:1px solid #2a2f3e;">
                    <td style="padding:9px 8px;color:#e8eaf0;">Decision Tree</td>
                    <td style="text-align:right;padding:9px 8px;color:#f59e0b;">9.79</td>
                    <td style="text-align:right;padding:9px 8px;color:#f59e0b;">0.4477</td>
                    <td style="text-align:center;padding:9px 8px;">
                        <span class="badge badge-ok">Medium</span></td>
                </tr>
                <tr style="border-bottom:1px solid #2a2f3e;">
                    <td style="padding:9px 8px;color:#e8eaf0;">Random Forest</td>
                    <td style="text-align:right;padding:9px 8px;color:#60a5fa;">9.49</td>
                    <td style="text-align:right;padding:9px 8px;color:#60a5fa;">0.4808</td>
                    <td style="text-align:center;padding:9px 8px;">
                        <span class="badge badge-good">Good</span></td>
                </tr>
                <tr style="border-bottom:1px solid #2a2f3e;">
                    <td style="padding:9px 8px;color:#e8eaf0;">XGBoost</td>
                    <td style="text-align:right;padding:9px 8px;color:#60a5fa;">8.31</td>
                    <td style="text-align:right;padding:9px 8px;color:#60a5fa;">0.6020</td>
                    <td style="text-align:center;padding:9px 8px;">
                        <span class="badge badge-best">Best ✓</span></td>
                </tr>
                <tr>
                    <td style="padding:9px 8px;color:#e8eaf0;font-weight:600;">Extra Trees</td>
                    <td style="text-align:right;padding:9px 8px;color:#4ade80;font-weight:600;">10.25</td>
                    <td style="text-align:right;padding:9px 8px;color:#4ade80;font-weight:600;">0.3950</td>
                    <td style="text-align:center;padding:9px 8px;">
                        <span class="badge badge-good">Medium</span></td>
                </tr>
            </tbody>
        </table>
        </div>""", unsafe_allow_html=True)

    with col2:
        # plt.rcParams.update(CHART_STYLE)
        # fig, ax = plt.subplots(figsize=(5, 4))
        # models_list = ['Linear\nRegression','Decision\nTree','Random\nForest',
        #                'XGBoost','Extra\nTrees']
        # r2_vals = [0.0956, 0.4477, 0.4808, 0.6020, 0.3950]
        # bar_colors = ['#f87171','#f59e0b','#60a5fa','#60a5fa','#4ade80']
        # bars = ax.barh(models_list, r2_vals, color=bar_colors, height=0.55)
        # for bar, val in zip(bars, r2_vals):
        #     ax.text(val+0.01, bar.get_y()+bar.get_height()/2,
        #             f'{val:.4f}', va='center', fontsize=8.5,
        #             color='#212529', fontweight='600')
        # ax.set_xlabel('R² Score', fontsize=9.5, color='#495057')
        # ax.set_xlim(0, 1.05)
        # ax.axvline(0.4, color=ACCENT, linestyle='--', alpha=0.6, linewidth=1.2)
        # ax.text(0.5, 4.4, 'R²=0.5', fontsize=7.5, color=ACCENT)
        # ax.set_title('Compare R² — 5 models', fontsize=10,
        #              fontweight='bold', color='#212529', pad=10)
        # ax.spines['top'].set_visible(False)
        # ax.spines['right'].set_visible(False)
        # ax.grid(True, axis='x', alpha=0.4)
        # fig.tight_layout()
        # # st.pyplot(fig)
        # st.plotly_chart(fig)
        # plt.close()
        _df = pd.DataFrame({
            'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'Extra Trees'],
            'R2 Score': [0.0956, 0.4477, 0.4808, 0.6020, 0.3950],
            'Color': ['#f87171', '#f59e0b', '#60a5fa', '#60a5fa', '#4ade80']
        })

        # 2. Vẽ biểu đồ bằng Plotly
        fig = px.bar(_df, 
                    x='R2 Score', 
                    y='Model', 
                    orientation='h',
                    text='R2 Score', # Hiển thị số trên cột
                    title='Compare R² — 5 models',
                    color='Color', 
                    color_discrete_map="identity") # Giữ đúng mã màu chủ nhân chọn

        # 3. Tinh chỉnh để đẹp như bản cũ
        fig.update_traces(texttemplate='%{text:.4f}', textposition='outside')
        fig.update_layout(
            xaxis_range=[0, 1.05],
            showlegend=False,
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=40, b=20),
            height=400
        )

        # Thêm đường kẻ phụ (Reference Line) tại R² = 0.5
        fig.add_vline(x=0.5, line_dash="dash", line_color="#495057", opacity=0.6)

        # 4. Hiển thị lên Streamlit
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">Key Findings</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    findings = [
        ("🏆 XGBoost as the Best Model",
         "R²=0.60 vs Linear R²=0.10. A nonlinear model that captures complex interactions among CVD risk factors."),
        ("📊 BMI & Cholesterol Dominate",
         "BMI contributes 36.71% and Cholesterol 35.61% in XGBoost — together accounting for over 72% of the model’s predictive power."),
        ("📈 Increasing Trend in CVD Rates",
         "CVD rates increased from 39.55 (2010) to 40.13 (2015), alongside a steady rise in global BMI."),
    ]
    for col, (title, text) in zip([col1, col2, col3], findings):
        with col:
            st.markdown(f"""
            <div class="insight-box">
                <strong>{title}</strong><br>{text}
            </div>""", unsafe_allow_html=True)

# =================================================================
# PAGE: EDA
# =================================================================
elif page == "🔍 EDA":
    # st.markdown('<div class="section-title">Descriptive statistics</div>', unsafe_allow_html=True)

    # 1. Tạo một dictionary để map tên cột từ tiếng Anh sang tiếng Việt (như trong ảnh)
    name_mapping = {
        'cardiovascular_diseases': 'CVD Rate',
        'air_pollution': 'Air pollution*',      # Thay bằng tên cột thực tế của bạn
        'alcohol_consumption': 'Alcohol consumption',    # Thay bằng tên cột thực tế của bạn
        'bmi': 'BMI',
        'cholesterol': 'Cholesterol',
        'diabetes': 'Diabetes',
        'glucose': 'Glucose',
        'physical_activity': 'Physical activities',
        'smoking': 'Tobacco'
    }

    # 2. Lấy dữ liệu thống kê và xử lý
    # Lưu ý: Ta dùng .loc để lấy đúng thứ tự các biến muốn hiển thị
    features_to_show = [f for f in feature_names if f in name_mapping] + ['cardiovascular_diseases']
    desc = df[features_to_show].describe().T[['mean', 'std', 'min', 'max']]

    # 3. Đổi tên các chỉ số thống kê (Index của cột)
    desc.columns = ['TB', 'Std', 'Min', 'Max']

    # 4. Đổi tên các hàng (Biến) theo mapping đã tạo
    desc.index = desc.index.map(lambda x: name_mapping.get(x, x))

    # 5. Hiển thị lên Streamlit
    st.markdown('<div class="section-title">Descriptive Statistics</div>', unsafe_allow_html=True)
    st.dataframe(
        desc.round(2).style.background_gradient(cmap='Blues', subset=['TB']),
        use_container_width=True
    )

    st.markdown('<div class="section-title">Missing Value Analysis</div>', unsafe_allow_html=True)

    miss_data = pd.DataFrame({
        'Đặc trưng': ['infrastructure (dropped)','alcohol_consumption',
                      'air_pollution','BMI','cholesterol','diabetes',
                      'glucose','physical_activities','tobacco'],
        'Thiếu':     [8446, 221, 66, 0, 0, 0, 0, 0, 0],
        'Tỷ lệ (%)': [77.09, 2.02, 0.60, 0, 0, 0, 0, 0, 0]
    })

    col1, col2 = st.columns([2, 1])
    with col1:
        # plt.rcParams.update(CHART_STYLE)
        # fig, ax = plt.subplots(figsize=(8, 4))
        # miss_sorted = miss_data.sort_values('Tỷ lệ (%)', ascending=True)
        # bar_c = [ACCENT if x > 50 else '#4361EE' if x > 0 else '#DEE2E6'
        #          for x in miss_sorted['Tỷ lệ (%)']]
        # bars = ax.barh(miss_sorted['Đặc trưng'], miss_sorted['Tỷ lệ (%)'],
        #                color=bar_c, height=0.6, edgecolor='white')
        # for bar, val in zip(bars, miss_sorted['Tỷ lệ (%)']):
        #     if val > 0:
        #         ax.text(val+0.5, bar.get_y()+bar.get_height()/2,
        #                 f'{val}%', va='center', fontsize=9, color='#212529', fontweight='600')
        # ax.set_xlabel('Missing Rate (%)', fontsize=10)
        # ax.set_title(' Missing value by feature', fontsize=11,
        #              fontweight='bold', color='#212529', pad=10)
        # ax.axvline(50, color=ACCENT, linestyle='--', alpha=0.5, linewidth=1.2)
        # ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        # ax.grid(True, axis='x', alpha=0.4)
        # fig.tight_layout()
        # # st.pyplot(fig); 
        # st.plotly_chart(fig)
        # plt.close()
        # 1. Sắp xếp dữ liệu (giống như miss_sorted của chủ nhân)
        miss_sorted = miss_data.sort_values('Tỷ lệ (%)', ascending=True)

        # 2. Xác định màu sắc dựa trên điều kiện (giống logic bar_c)
        # Chúng ta thêm một cột 'Color' vào DataFrame để Plotly hiểu
        def get_color(x):
            if x > 50: return '#FF4B4B' # Giả định ACCENT là màu đỏ
            elif x > 0: return '#4361EE'
            else: return '#DEE2E6'

        miss_sorted['Color'] = miss_sorted['Tỷ lệ (%)'].apply(get_color)

        # 3. Vẽ biểu đồ bằng Plotly
        fig = px.bar(miss_sorted, 
                    x='Tỷ lệ (%)', 
                    y='Đặc trưng', 
                    orientation='h',
                    text='Tỷ lệ (%)', # Hiển thị con số %
                    title='Missing value by feature',
                    color='Color',
                    color_discrete_map="identity")

        # 4. Tinh chỉnh các thông số hiển thị (Trục, lưới, đường kẻ phụ)
        fig.update_traces(
            texttemplate='%{text}%', 
            textposition='outside',
            marker_line_color='white',
            marker_line_width=1
        )

        fig.update_layout(
            xaxis_title='Missing Rate (%)',
            yaxis_title=None,
            plot_bgcolor='white',
            xaxis=dict(range=[0, 105], showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
            title_font=dict(size=14, family="Arial", color="#212529"),
            showlegend=False,
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        # Thêm đường kẻ phụ (Threshold) tại mức 50%
        fig.add_vline(x=50, line_dash="dash", line_color="#FF4B4B", opacity=0.5)

        # 5. Hiển thị lên Streamlit
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""
        <div class="insight-box" style="margin-top:30px;">
            <strong>Decision to Exclude Infrastructure Features</strong><br><br>
            x7 thiếu <strong>77.09%</strong> (8.446/10.956 records) —
            Imputation would introduce significant bias.<br><br>
            Giải pháp: <strong>Completely excluded</strong> from the feature set.
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Preprocessing Pipeline</div>', unsafe_allow_html=True)
    steps = [
        ("01","Lọc 2010–2015","TimeDim BETWEEN 2010 AND 2015 → 10.956 bản ghi"),
        ("02","Remove Infrastructure Features","x7 has 77.09% missing values → excluded from the feature set"),
        ("03","Missing Value Imputation","air_pollution (0.60%) and alcohol (2.02%) → imputed using column mean"),
        ("04","Log transform","air_pollution → log1p() transformation to reduce right skewness"),
        ("05","Outlier Removal IQR×3","Applied simultaneously across all features → 8.605 cleaned records"),
        ("06","StandardScaler","Standardized to mean = 0 and standard deviation = 1 before training"),
    ]
    for num, title, desc in steps:
        st.markdown(f"""
        <div style='display:flex;gap:16px;align-items:flex-start;
                    padding:12px 0;border-bottom:1px solid #2a2f3e;'>
            <div style='font-family:IBM Plex Mono,monospace;font-size:0.7rem;
                        color:#e05c5c;min-width:28px;padding-top:2px;'>{num}</div>
            <div>
                <div style='font-weight:600;color:#e8eaf0;font-size:0.88rem;'>{title}</div>
                <div style='color:#8b90a0;font-size:0.8rem;margin-top:3px;'>{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

# =================================================================
# PAGE: VISUALIZATION
# =================================================================
elif page == "📈 Visualization":
    # plt.rcParams.update(CHART_STYLE)

    # # Correlation Heatmap
    # st.markdown('<div class="section-title">Correlation matrix</div>', unsafe_allow_html=True)
    # features_all = feature_names + ['cardiovascular_diseases']
    # df_scaled = df.copy()
    # sc = StandardScaler()
    # df_scaled[feature_names] = sc.fit_transform(df[feature_names])
    # corr_matrix = df_scaled[features_all].corr()

    # fig, ax = plt.subplots(figsize=(10, 7))
    # sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', fmt='.2f',
    #             linewidths=0.5, linecolor='#F8F9FA', ax=ax,
    #             annot_kws={'size': 8}, vmin=-1, vmax=1,
    #             cbar_kws={'shrink': 0.8})
    # ax.set_title('Correlation Matrix — Risk Factors and CVD Rate',
    #              fontsize=11, fontweight='bold', color='#212529', pad=12)
    # ax.tick_params(axis='x', rotation=45, labelsize=8)
    # ax.tick_params(axis='y', rotation=0, labelsize=8)
    # fig.tight_layout()
    # # st.pyplot(fig)
    # st.plotly_chart(fig)
    # plt.close()
    # 1. Tính toán ma trận tương quan (giữ nguyên logic của chủ nhân)
    features_all = feature_names + ['cardiovascular_diseases']
    corr_matrix = df[features_all].corr() # Lưu ý: Tương quan không đổi khi dùng StandardScaler nên có thể dùng trực tiếp df

    # 2. Chuẩn bị dữ liệu cho Plotly
    z = corr_matrix.values
    x = list(corr_matrix.columns)
    y = list(corr_matrix.index)

    # 3. Tạo Heatmap bằng Figure Factory (để dễ dàng hiển thị con số 'annot')
    fig = ff.create_annotated_heatmap(
        z=z,
        x=x,
        y=y,
        annotation_text=np.around(z, decimals=2), # Làm tròn 2 chữ số (fmt='.2f')
        colorscale='redor', # Bảng màu tương tự 'RdBu_r'
        zmin=-1, zmax=1,
        showscale=True
    )

    # 4. Tinh chỉnh giao diện cho "sang chảnh"
    fig.update_layout(
        title='Correlation Matrix — Risk Factors and CVD Rate',
        title_x=0.5, # Căn giữa tiêu đề
        title_font=dict(size=14, color='#212529', family="Arial Black"),
        margin=dict(t=100, l=150), # Chừa khoảng trống cho nhãn trục Y
        xaxis=dict(side='bottom'), # Đưa nhãn trục X xuống dưới
        width=800,
        height=700
    )

    # Tinh chỉnh font chữ cho các con số bên trong ô
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 8

    # 5. Hiển thị lên Streamlit
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <strong>Insight:</strong>Most feature pairs exhibit low correlation (−0.14 đến 0.39),
        reduced multicollinearity. holesterol shows the strongest positive correlation with CVD.
        Air pollution and diabetes exhibit r≈0.39 — indicating a combined environmental–metabolic burden.
    </div>""", unsafe_allow_html=True)

    # CVD Trend
    st.markdown('<div class="section-title">CVD Trend Over Time</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        from PIL import Image
        img1 = Image.open("CVD_trend.png")
        st.image(img1, use_container_width=True)
    with col2:
        st.markdown("""
        <div class="insight-box" style="margin-top:20px;">
            <strong>Sustainable Upward Trend</strong><br><br>
            Cardiovascular Disease (CVD) increased from <strong>39.55</strong> in 2010 to <strong>40.13</strong> in 2015,
            reflecting the gradual accumulation of global risk factors..
        </div>""", unsafe_allow_html=True)

    # Top 10 Countries
    st.markdown('<div class="section-title">Top 10 Countries with the Highest CVD</div>', unsafe_allow_html=True)
    conn_viz = sqlite3.connect("sample_v4.db")
    df_top = pd.read_sql("""
        SELECT SpatialDim AS country, ROUND(AVG(y),2) AS avg_cvd
        FROM NearsestSample
        WHERE TimeDim BETWEEN 2010 AND 2015 AND y IS NOT NULL
        GROUP BY SpatialDim ORDER BY avg_cvd DESC LIMIT 10
    """, conn_viz); conn_viz.close()
    country_names = {
        'LTU':'Lithuania','HRV':'Croatia','BLR':'Belarus','PRY':'Paraguay',
        'RUS':'Russia','KAZ':'Kazakhstan','MDA':'Moldova',
        'LVA':'Latvia','MNG':'Mongolia','ROU':'Romania'
    }
    df_top['label'] = df_top['country'].map(lambda x: country_names.get(x, x))
    df_top = df_top.sort_values('avg_cvd', ascending=True)

    # fig, ax = plt.subplots(figsize=(10, 5.5))
    # norm = plt.Normalize(df_top['avg_cvd'].min(), df_top['avg_cvd'].max())
    # bar_colors = [plt.cm.RdPu(norm(v)*0.7+0.25) for v in df_top['avg_cvd']]
    # bars = ax.barh(df_top['label'], df_top['avg_cvd'], color=bar_colors,
    #                height=0.62, edgecolor='white', linewidth=0.5)
    # for bar, val in zip(bars, df_top['avg_cvd']):
    #     ax.text(val+0.3, bar.get_y()+bar.get_height()/2, f'{val:.2f}',
    #             va='center', ha='left', fontsize=10, fontweight='600', color='#212529')
    # ax.axvline(39.86, color=ACCENT, linestyle='--', linewidth=1.8, alpha=0.8)
    # ax.text(39.86+0.4, 0.3, f'Global average\n39.86', fontsize=8.5,
    #         color=ACCENT, va='bottom', fontweight='600')
    # ax.set_xlabel('Average CVD', fontsize=11)
    # ax.set_title('Top 10 Countries with the Highest CVD Rates (2010–2015)',
    #              fontsize=11, fontweight='bold', color='#212529', pad=12)
    # ax.set_xlim(0, 80)
    # ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    # ax.grid(True, axis='x', alpha=0.4)
    # fig.tight_layout()
    # # st.pyplot(fig)
    # st.plotly_chart(fig)
    # plt.close()
    fig = px.bar(df_top, 
             x='avg_cvd', 
             y='label', 
             orientation='h',
             text='avg_cvd',
             title='Top 10 Countries with the Highest CVD Rates (2010–2015)',
             color='avg_cvd',
             color_continuous_scale='RdPu') # Dải màu RdPu giống Matplotlib

    # 3. Tinh chỉnh đường nét và con số
    fig.update_traces(
        texttemplate='%{text:.2f}', 
        textposition='outside',
        marker_line_color='white',
        marker_line_width=0.5,
        hovertemplate='<b>%{y}</b><br>Average CVD: %{x:.2f}' # Nội dung hiện khi rê chuột
    )

    # 4. Thêm đường trung bình toàn cầu (Global Average Line)
    global_avg = 39.86
    fig.add_vline(x=global_avg, 
                line_dash="dash", 
                line_color="#FF4B4B", # Màu ACCENT của chủ nhân
                line_width=2,
                annotation_text=f"Global average: {global_avg}", 
                annotation_position="bottom right")

    # 5. Thiết lập bố cục (Layout)
    fig.update_layout(
        xaxis_title='Average CVD',
        yaxis_title=None,
        xaxis=dict(range=[0, 80], showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
        plot_bgcolor='white',
        coloraxis_showscale=False, # Ẩn thanh màu bên cạnh nếu không cần thiết
        margin=dict(l=20, r=20, t=60, b=20),
        height=500
    )

    # 6. Hiển thị lên Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Feature Importance — RF & XGBoost side by side (bar chart)
    st.markdown('<div class="section-title">Characteristic Importance</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    for col, imp, title, cmap_name in [
        (col1, rf_imp,  'Random Forest', 'Blues'),
        (col2, xgb_imp, 'XGBoost',       'RdPu'),
    ]:
        with col:
            # fig, ax = plt.subplots(figsize=(6, 4.5))
            # sorted_idx = np.argsort(imp)
            # sorted_imp = [imp[i] for i in sorted_idx]
            # sorted_lbl = [LABELS_VI[i] for i in sorted_idx]
            # cmap = plt.get_cmap(cmap_name)
            # norm2 = plt.Normalize(min(sorted_imp), max(sorted_imp))
            # bc = [cmap(norm2(v)*0.7+0.25) for v in sorted_imp]
            # bars = ax.barh(sorted_lbl, sorted_imp, color=bc, height=0.6,
            #                edgecolor='white', linewidth=0.5)
            # for bar, val in zip(bars, sorted_imp):
            #     ax.text(val+0.003, bar.get_y()+bar.get_height()/2,
            #             f'{val*100:.2f}%', va='center', ha='left',
            #             fontsize=9, fontweight='600', color='#212529')
            # ax.set_xlabel('Feature Importance', fontsize=10)
            # ax.set_title(f'Feature Importance\n({title})', fontsize=10.5,
            #              fontweight='bold', color='#212529', pad=10)
            # ax.set_xlim(0, max(sorted_imp)*1.2)
            # ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            # ax.grid(True, axis='x', alpha=0.4)
            # fig.tight_layout()
            # # st.pyplot(fig)
            # st.plotly_chart(fig)
            # plt.close()
            # 1. Chuẩn bị dữ liệu (Sắp xếp lại giống logic sorted_idx của chủ nhân)
# Giả sử imp và LABELS_VI là các list dữ liệu của chủ nhân
            df_imp = pd.DataFrame({
                'Feature': LABELS_VI,
                'Importance': imp
            }).sort_values('Importance', ascending=True)

            # 2. Vẽ biểu đồ bằng Plotly
            fig = px.bar(df_imp, 
                        x='Importance', 
                        y='Feature', 
                        orientation='h',
                        text='Importance', # Hiển thị giá trị phần trăm
                        title=f'Feature Importance<br>({title})',
                        color='Importance',
                        color_continuous_scale='Viridis') # Hoặc dùng cmap_name nếu Plotly có hỗ trợ

            # 3. Tinh chỉnh định dạng số và nhãn
            fig.update_traces(
                texttemplate='%{text:.2%}', # Chuyển số thập phân thành dạng 12.34%
                textposition='outside',
                marker_line_color='white',
                marker_line_width=0.5,
                hovertemplate='<b>%{y}</b><br>Feature importance: %{x:.2%}'
            )

            # 4. Thiết lập bố cục (Layout)
            fig.update_layout(
                xaxis_title='Feature Importance',
                yaxis_title=None,
                xaxis=dict(range=[0, max(imp)*1.2], showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
                plot_bgcolor='black',
                coloraxis_showscale=False, # Ẩn thanh màu bên cạnh
                title_font=dict(size=14, color='#212529', family="Arial Black"),
                margin=dict(l=20, r=20, t=60, b=20),
                height=500
            )

            # 5. Hiển thị lên Streamlit
            st.plotly_chart(fig, use_container_width=True)

    # Factor Groups pie
    st.markdown('<div class="section-title">Factor Group Analysis</div>', unsafe_allow_html=True)

    groups_def = {
        'Lifestyle\n(BMI, Tobacco, Alcohol consumption, Physical activities)':
            ['physical_activities','tobacco','alcohol_consumption','BMI'],
        'Environment\n(Air pollution)': ['air_pollution'],
        'Clinial factors\n(Cholesterol, Diabetes, Glucose)':
            ['cholesterol','diabetes','glucose'],
    }
    feat_map = {f: i for i, f in enumerate(feature_names)}
    group_vals = {}
    for name, comps in groups_def.items():
        group_vals[name] = sum(xgb_imp[feat_map[c]] for c in comps)

    col1, col2 = st.columns([1, 2])
    with col1:
        # fig, ax = plt.subplots(figsize=(5, 5))
        # wedges, _, autotexts = ax.pie(
        #     group_vals.values(), labels=None,
        #     autopct='%1.2f%%', colors=['#4361EE','#F72585','#7209B7'],
        #     startangle=90, pctdistance=0.68,
        #     wedgeprops=dict(linewidth=2.5, edgecolor='#F8F9FA')
        # )
        # for at in autotexts:
        #     at.set_fontsize(11); at.set_fontweight('bold'); at.set_color('white')
        # ax.legend(list(group_vals.keys()), loc='lower center',
        #           bbox_to_anchor=(0.5, -0.3), ncol=1, fontsize=9,
        #           framealpha=0, labelcolor='#212529')
        # ax.set_title('Contribution by Group \n(XGBoost)', fontsize=10.5,
        #              fontweight='bold', color='#212529', pad=14)
        # fig.tight_layout()
        # # st.pyplot(fig)
        # st.plotly_chart(fig)
        # plt.close()
        # 1. Chuẩn bị dữ liệu từ dictionary group_vals
        labels = list(group_vals.keys())
        values = list(group_vals.values())

        # 2. Vẽ biểu đồ bằng Plotly
        fig = px.pie(
            names=labels, 
            values=values,
            title='Contribution by Group<br>(XGBoost)',
            color_discrete_sequence=['#4361EE', '#F72585', '#7209B7'] # Giữ đúng bộ màu của chủ nhân
        )

        # 3. Tinh chỉnh hiển thị (giống các thiết lập wedges và autotexts)
        fig.update_traces(
            textinfo='percent', # Chỉ hiện phần trăm (giống autopct='%1.2f%%')
            textfont_size=14,
            textfont_color='white',
            marker=dict(line=dict(color='#F8F9FA', width=2.5)), # Giống wedgeprops
            hovertemplate="<b>%{label}</b><br>Tỷ lệ: %{percent:.2%}<br>Giá trị: %{value}<extra></extra>"
        )

        # 4. Thiết lập bố cục (Layout)
        fig.update_layout(
            title_font=dict(size=14, color='#212529', family="Arial Black"),
            legend=dict(
                orientation="h",       # Chú thích nằm ngang
                yanchor="bottom",
                y=-0.2,                # Đưa chú thích xuống dưới giống bbox_to_anchor
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=80, b=50, l=20, r=20),
            width=500,
            height=550
        )

        # 5. Hiển thị lên Streamlit
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        for group, val in group_vals.items():
            name = group.replace('\n', ' ')
            pct = val * 100
            color = '#4361EE' if 'Lifestyle' in name else \
                    '#F72585' if 'Environment' in name else '#7209B7'
            st.markdown(f"""
            <div style='margin:14px 0;'>
                <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
                    <span style='font-size:0.85rem;color:#e8eaf0;'>{name}</span>
                    <span style='font-family:IBM Plex Mono,monospace;font-size:0.85rem;
                                color:{color};font-weight:600;'>{pct:.2f}%</span>
                </div>
                <div style='height:10px;background:#2a2f3e;border-radius:5px;'>
                    <div style='height:100%;width:{pct:.1f}%;background:{color};
                                border-radius:5px;'></div>
                </div>
            </div>""", unsafe_allow_html=True)

# =================================================================
# PAGE: MODELING
# =================================================================
elif page == "🤖 Modeling":
    plt.rcParams.update(CHART_STYLE)

    st.markdown('<div class="section-title">Model Configuration</div>', unsafe_allow_html=True)

    configs = [
        ("Linear Regression","Baseline",
         "Solver: OLS\nStatsmodels: Test Statistic\nFeatures: 8","#f87171"),
        ("Decision Tree","Tree",
         "max_depth: 5\nmin_samples_split: 10\nmin_samples_leaf: 5","#f59e0b"),
        ("Extra Trees","Ensemble",
         "n_estimators: 200\nmax_depth: 5\nrandom_state: 42","#a78bfa"),
        ("Random Forest","Ensemble",
         "n_estimators: 200\nmax_depth: 5\nrandom_state: 42","#60a5fa"),
        ("XGBoost","Gradient Boosting",
         "n_estimators: 300\nmax_depth: 4\nlr: 0.05 | subsample: 0.8","#4ade80"),
    ]
    cols_cfg = st.columns(5)
    for col, (name, kind, cfg, color) in zip(cols_cfg, configs):
        with col:
            st.markdown(f"""
            <div class="chart-container" style="border-left:3px solid {color};">
                <div style='font-family:IBM Plex Mono,monospace;font-size:0.65rem;
                            color:{color};letter-spacing:1px;'>{kind.upper()}</div>
                <div style='font-size:0.9rem;font-weight:600;color:#e8eaf0;margin:6px 0;'>{name}</div>
                <div style='font-family:IBM Plex Mono,monospace;font-size:0.68rem;
                            color:#8b90a0;white-space:pre-line;'>{cfg}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Model performance</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    models_list = ['Linear Regression','Decision Tree','Random Forest','XGBoost','Extra Trees']
    mse_vals    = [12.53, 9.79, 9.49, 8.31, 10.25]
    r2_vals     = [0.0956,  0.4477, 0.4808, 0.6020, 0.3950]
    bar_colors  = ['#f87171','#f59e0b','#60a5fa','#60a5fa','#4ade80']

    with col1:
        # fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
        # # R²
        # axes[0].barh(models_list, r2_vals, color=bar_colors, height=0.55)
        # axes[0].set_xlabel('R²', fontsize=9.5); axes[0].set_xlim(0, 1.05)
        # axes[0].axvline(0.8, color=ACCENT, linestyle='--', alpha=0.5, lw=1.2)
        # axes[0].set_title('R² Score', fontsize=10, fontweight='bold', color='#212529')
        # for i, v in enumerate(r2_vals):
        #     axes[0].text(v+0.01, i, f'{v:.4f}', va='center', fontsize=8, color='#212529')
        # axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)
        # axes[0].grid(True, axis='x', alpha=0.4)
        # # MSE
        # axes[1].barh(models_list, mse_vals, color=bar_colors, height=0.55)
        # axes[1].set_xlabel('RMSE', fontsize=9.5)
        # axes[1].set_title('Root Mean Squared Error', fontsize=10, fontweight='bold', color='#212529')
        # for i, v in enumerate(mse_vals):
        #     axes[1].text(v+5, i, f'{v:.1f}', va='center', fontsize=8, color='#212529')
        # axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)
        # axes[1].grid(True, axis='x', alpha=0.4)
        # fig.tight_layout()
        # # st.pyplot(fig)
        # st.plotly_chart(fig)
        # plt.close()
        # 1. Khởi tạo Subplots với 1 hàng và 2 cột
        fig = make_subplots(rows=1, cols=2, 
                            subplot_titles=("R² Score", "Root Mean Squared Error"),
                            horizontal_spacing=0.15)

        # 2. Thêm biểu đồ R² vào cột 1
        fig.add_trace(
            go.Bar(
                x=r2_vals,
                y=models_list,
                orientation='h',
                marker_color=bar_colors,
                text=[f'{v:.4f}' for v in r2_vals],
                textposition='outside',
                name='R²'
            ),
            row=1, col=1
        )

        # 3. Thêm biểu đồ RMSE vào cột 2
        fig.add_trace(
            go.Bar(
                x=mse_vals,
                y=models_list,
                orientation='h',
                marker_color=bar_colors,
                text=[f'{v:.1f}' for v in mse_vals],
                textposition='outside',
                name='RMSE'
            ),
            row=1, col=2
        )

        # 4. Tinh chỉnh Layout và các đường kẻ phụ (Reference lines)
        fig.update_layout(
            height=450,
            showlegend=False,
            plot_bgcolor='white',
            margin=dict(t=80, b=40, l=20, r=20)
        )

        # Thêm đường kẻ nét đứt R² = 0.8 tại cột 1
        fig.add_vline(x=0.8, line_dash="dash", line_color=ACCENT, opacity=0.5, row=1, col=1)

        # Định dạng trục X cho cả hai biểu đồ
        fig.update_xaxes(title_text="R²", range=[0, 1.1], row=1, col=1, gridcolor='rgba(0,0,0,0.1)')
        fig.update_xaxes(title_text="RMSE", row=1, col=2, gridcolor='rgba(0,0,0,0.1)')

        # 5. Hiển thị lên Streamlit
        st.plotly_chart(fig, use_container_width=True)

    with col2:
    # Hiển thị ảnh Actual vs Predicted — XGBoost
        from PIL import Image

        img = Image.open("XGBoost.png")
        st.image(img, caption="Actual vs Predicted — XGBoost", use_container_width=True)

    # OLS Summary
    st.markdown('<div class="section-title">Result of OLS (statsmodels)</div>', unsafe_allow_html=True)
    ols_data = pd.DataFrame({
        'Feature':  ['const','air_pollution','alcohol_consumption','BMI','cholesterol',
                       'diabetes','glucose','physical_activities','tobacco'],
        'Coef':      [39.86, 0.18, -2.15, -1.12, 2.50, -1.48, 2.01, 0.73, -0.38],
        'Std Err':    [0.134, 0.141, 0.137, 0.145, 0.142, 0.144, 0.153, 0.148, 0.138],
        't-stat':     [297.46, 1.28, -15.69, -7.72, 17.61, -10.28, 13.14, 4.93, -2.75],
        'p-value':    ['<0.001','0.201','<0.001','<0.001','<0.001','<0.001','<0.001','<0.001','0.006'],
        'Meaning':    ['✅','❌','✅','✅','✅','✅','✅','✅','✅']
    })
    st.dataframe(ols_data.style.background_gradient(cmap='Blues', subset=['t-stat']),
                 use_container_width=True)
    st.markdown("""
    <div class="insight-box">
        <strong>OLS:</strong> R²=0.185, F=298.4 (p≈0.000).
        <strong>7/8 features are statistically significant</strong> (p&lt;0.05) —
        seperate <strong>air_pollution</strong> below the threshold (p=0.201).
        <strong>Cholesterol</strong> shows the greatest positive coefficient (+2.50) —
        has the most significant positive linear relationship with CVD Rate.
    </div>""", unsafe_allow_html=True)

# =================================================================
# PAGE: SQL ANALYSIS
# =================================================================
elif page == "🗄️ SQL Analysis":
    st.markdown('<div class="section-title">SQL Queries & Result</div>', unsafe_allow_html=True)

    try:
        conn_sql = sqlite3.connect("sample_v4.db")
    except Exception as e:
        st.error(f"Can not connect to DB: {e}"); st.stop()

    queries_display = [
        {
            "title": "Q1 — Annual CVD Trends (2010–2015)",
            "question": "How has the CVD rate changed over the years?",
            "sql": """SELECT TimeDim AS year, COUNT(*) AS total_records,
  ROUND(AVG(y),2) AS avg_cvd, ROUND(MIN(y),2) AS min_cvd, ROUND(MAX(y),2) AS max_cvd
FROM NearsestSample
WHERE TimeDim BETWEEN 2010 AND 2015
GROUP BY TimeDim ORDER BY TimeDim""",
        },
        {
            "title": "Q2 — Top 10 Highest CVD Prevalence Countries",
            "question": "Which country carries the greatest CVD burden?",
            "sql": """SELECT SpatialDim AS country, ROUND(AVG(y),2) AS avg_cvd, COUNT(*) AS records
FROM NearsestSample
WHERE TimeDim BETWEEN 2010 AND 2015 AND y IS NOT NULL
GROUP BY SpatialDim ORDER BY avg_cvd DESC LIMIT 10""",
        },
        {
            "title": "Q3 — Average Risk Factors by Year",
            "question": "How do metabolic factors change over time?",
            "sql": """SELECT TimeDim AS year,
  ROUND(AVG(x3),2) AS avg_BMI, ROUND(AVG(x4),2) AS avg_cholesterol,
  ROUND(AVG(x5),2) AS avg_diabetes, ROUND(AVG(x6),2) AS avg_glucose,
  ROUND(AVG(x8),2) AS avg_physical_activities, ROUND(AVG(x9),2) AS avg_tobacco
FROM NearsestSample
WHERE TimeDim BETWEEN 2010 AND 2015
GROUP BY TimeDim ORDER BY TimeDim""",
        },
        {
            "title": "Q4 — Analysis of Missing Data",
            "question": "What is the missing rate for each feature, and how does it justify the infrastructure type?",
            "sql": """SELECT 'infrastructure(dropped)' AS feature,
    COUNT(*)-COUNT(x7) AS missing,
    ROUND(100.0*(COUNT(*)-COUNT(x7))/COUNT(*),2) AS pct_missing
FROM NearsestSample WHERE TimeDim BETWEEN 2010 AND 2015
UNION ALL SELECT 'air_pollution', COUNT(*)-COUNT(x1),
    ROUND(100.0*(COUNT(*)-COUNT(x1))/COUNT(*),2)
FROM NearsestSample WHERE TimeDim BETWEEN 2010 AND 2015
UNION ALL SELECT 'alcohol_consumption', COUNT(*)-COUNT(x2),
    ROUND(100.0*(COUNT(*)-COUNT(x2))/COUNT(*),2)
FROM NearsestSample WHERE TimeDim BETWEEN 2010 AND 2015
ORDER BY pct_missing DESC""",
        },
        {
            "title": "Q5 — Descriptive Statistics for Variables",
            "question": "What are the descriptive statistics for each risk factor?",
            "sql": """SELECT 'cardiovascular_diseases' AS variable,
    ROUND(AVG(y),2) AS mean, ROUND(MIN(y),2) AS min, ROUND(MAX(y),2) AS max
FROM NearsestSample WHERE TimeDim BETWEEN 2010 AND 2015
UNION ALL SELECT 'BMI', ROUND(AVG(x3),2), ROUND(MIN(x3),2), ROUND(MAX(x3),2)
FROM NearsestSample WHERE TimeDim BETWEEN 2010 AND 2015
UNION ALL SELECT 'cholesterol', ROUND(AVG(x4),2), ROUND(MIN(x4),2), ROUND(MAX(x4),2)
FROM NearsestSample WHERE TimeDim BETWEEN 2010 AND 2015
UNION ALL SELECT 'diabetes', ROUND(AVG(x5),2), ROUND(MIN(x5),2), ROUND(MAX(x5),2)
FROM NearsestSample WHERE TimeDim BETWEEN 2010 AND 2015
UNION ALL SELECT 'tobacco', ROUND(AVG(x9),2), ROUND(MIN(x9),2), ROUND(MAX(x9),2)
FROM NearsestSample WHERE TimeDim BETWEEN 2010 AND 2015""",
        },
        {
            "title": "Q6 — Country Classification by CVD Levels",
            "question": "How many countries fall into the Low, Medium, or High CVD risk groups?",
            "sql": """SELECT
    CASE WHEN avg_cvd <= 30 THEN 'Low (CVD ≤ 30)'
         WHEN avg_cvd <= 70 THEN 'Medium (30 < CVD ≤ 70)'
         ELSE 'High (CVD > 70)'
    END AS nhom_cvd,
    COUNT(*) AS so_quoc_gia,
    ROUND(AVG(avg_cvd),2) AS tb_cvd_nhom
FROM (
    SELECT SpatialDim, AVG(y) AS avg_cvd
    FROM NearsestSample
    WHERE TimeDim BETWEEN 2010 AND 2015 AND y IS NOT NULL
    GROUP BY SpatialDim
)
GROUP BY nhom_cvd ORDER BY tb_cvd_nhom""",
        },
    ]

    for q in queries_display:
        with st.expander(f"**{q['title']}**", expanded=False):
            st.markdown(
                f"<div style='color:#8b90a0;font-size:0.82rem;margin-bottom:10px;'>"
                f"❓ {q['question']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='sql-block'>{q['sql']}</div>", unsafe_allow_html=True)
            try:
                result_df = pd.read_sql(q['sql'], conn_sql)
                st.dataframe(result_df, use_container_width=True)
            except Exception as e:
                st.error(f"Query error: {e}")

    conn_sql.close()

#### Project architecture
elif page == "🏗️ Project Architecture":
    st.markdown('<div class="section-title">End-to-End Pipeline Architecture</div>', unsafe_allow_html=True)
    
    # Sử dụng Graphviz để vẽ sơ đồ luồng dữ liệu (Data Flow)
    st.graphviz_chart('''
        digraph {
            node [shape=box, style=filled, color="#1a1f2e", fontcolor="#e8eaf0", fontname="IBM Plex Sans"]
            edge [color="#e05c5c"]
            
            subgraph cluster_0 {
                label = "Data Source";
                color="#2a2f3e";
                fontcolor="#8b90a0";
                "WHO GHO API" -> "SQLite DB (sample_v4.db)"
            }
            
            subgraph cluster_1 {
                label = "Preprocessing (Cleaning)";
                color="#2a2f3e";
                fontcolor="#8b90a0";
                "SQLite DB (sample_v4.db)" -> "Missing Imputation"
                "Missing Imputation" -> "Log Transform"
                "Log Transform" -> "IQR Outlier Removal"
                "IQR Outlier Removal" -> "Standard Scaling"
            }
            
            subgraph cluster_2 {
                label = "Modeling & Evaluation";
                color="#2a2f3e";
                fontcolor="#8b90a0";
                "Standard Scaling" -> "Linear Regression"
                "Standard Scaling" -> "Tree-based Models"
                "Tree-based Models" -> "XGBoost (Champion)"
            }
            
            "XGBoost (Champion)" -> "Streamlit Dashboard" [label="Inference"]
        }
    ''')

    st.markdown("""
    <div class="insight-box">
        <strong>Thuật ngữ hệ thống (System Terms):</strong><br>
        * <strong>Data Pipeline</strong>: Quy trình tự động hóa việc thu thập và xử lý dữ liệu.<br>
        * <strong>Inference Engine</strong>: Bộ máy sử dụng mô hình đã huấn luyện để đưa ra dự báo thực tế.
    </div>""", unsafe_allow_html=True)

# Sơ đồ Live Predictor
# elif page == "🔮 Live Predictor":
#     st.markdown('<div class="section-title">Interactive CVD Risk Prediction</div>', unsafe_allow_html=True)
    
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.markdown("### 🛠️ Input Health Metrics")
#         # Tạo các thanh trượt nhập liệu dựa trên đặc trưng của mô hình
#         val_bmi = st.slider("Chỉ số BMI (Body Mass Index)", 15.0, 100.0, 24.0)
#         val_chol = st.slider("Cholesterol (mmol/L)", 2.0, 8.0, 5.0)
#         val_glu = st.slider("Glucose (Huyết đường)", 3.0, 15.0, 5.5)
#         val_diabete = st.slider("Diabete", 0, 66, 1)
#         val_smoke = st.slider("Sử dụng thuốc lá (Tobacco)", 0, 100, 20)
#         val_physic = st.slider("Physical activity", 0, 100, 20)
#         val_alc = st.slider("Tiêu thụ rượu bia (Alcohol)", 0.0, 20.0, 5.0)
#         val_poll = st.slider("Ô nhiễm không khí (Air Pollution)", 0.0, 100.0, 25.0)

#     with col2:
#         st.markdown("### 📈 Prediction Result")
#         # Giả lập logic dự báo (Chủ nhân có thể thay bằng model.predict thực tế)
#         # Lưu ý: Trong thực tế cần StandardScaler.transform() dữ liệu này trước
#         # base_risk = 35.0
#         # calculated_risk = base_risk + (val_bmi * 0.2) + (val_chol * 0.5) + (val_smoke * 5.0)
#         calculated_risk = 0
#         xgboost = models["XGBoost"]
        
#         # feature_names = ['air_pollution','alcohol_consumption','BMI','cholesterol',
#         #              'diabetes','glucose','physical_activities','tobacco']
#         X_test = np.array([[val_poll, val_smoke, val_bmi, val_chol, val_diabete, val_glu, val_physic, val_smoke]])
#         predict = xgboost.predict(X_test)
#         calculated_risk = predict[0]
#         print(X_test, calculated_risk)

#         # Hiển thị kết quả bằng metric card
#         st.markdown(f"""
#         <div class="metric-card" style="background: #1e2536; border: 2px solid #e05c5c;">
#             <div class="metric-label">Estimated CVD Risk Score</div>
#             <div class="metric-value" style="color:#e05c5c;">{calculated_risk:.2f}%</div>
#             <div class="metric-sub">Dựa trên thuật toán XGBoost</div>
#         </div>
#         """, unsafe_allow_html=True)
        
#         if calculated_risk > 50:
#             st.warning("⚠️ Cảnh báo: Nguy cơ mắc bệnh tim mạch ở mức Cao.")
#         else:
#             st.success("✅ Tuyệt vời: Các chỉ số đang ở ngưỡng an toàn.")