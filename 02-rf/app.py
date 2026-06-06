"""
商品智能分类系统 — Streamlit Dashboard
随机森林 + TF-IDF 驱动，支持单条/批量预测 + 数据看板
"""
import os
import sys
import time
from datetime import datetime
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from rf_predict_fun import load_model, load_class_names, predict_single, predict_batch, predict_with_proba
from config import Config

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title='AI 商品分类系统',
    page_icon='🤖',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ============================================================
# 自定义 CSS
# ============================================================
st.markdown("""
<style>
    /* 全局 */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* 主背景 */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* 卡片 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.3);
        margin-bottom: 12px;
    }
    .metric-card.green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 4px 20px rgba(17, 153, 142, 0.3);
    }
    .metric-card.orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 4px 20px rgba(245, 87, 108, 0.3);
    }
    .metric-card.blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        box-shadow: 0 4px 20px rgba(79, 172, 254, 0.3);
    }
    .metric-card .label {
        font-size: 0.85rem;
        opacity: 0.85;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        font-size: 2rem;
        font-weight: 700;
        margin: 4px 0;
    }
    .metric-card .sub {
        font-size: 0.8rem;
        opacity: 0.7;
    }

    /* 预测结果卡片 */
    .predict-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 20px;
        padding: 32px;
        color: white;
        margin: 16px 0;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .predict-card .pred-label {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .predict-card .conf-bar {
        height: 8px;
        border-radius: 4px;
        margin: 8px 0;
        transition: width 0.5s ease;
    }

    /* 标签 Badge */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge.high { background: #d4edda; color: #155724; }
    .badge.medium { background: #fff3cd; color: #856404; }
    .badge.low { background: #f8d7da; color: #721c24; }

    /* 搜索框 */
    .search-box input {
        border-radius: 12px !important;
        border: 2px solid #e0e0e0 !important;
        padding: 12px 16px !important;
        font-size: 1rem !important;
        transition: all 0.3s !important;
    }
    .search-box input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102,126,234,0.2) !important;
    }

    /* 表格优化 */
    .dataframe {
        border-radius: 12px !important;
        overflow: hidden !important;
    }

    /* 侧边栏 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: rgba(255,255,255,0.9);
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: white !important;
    }

    /* 标题动画 */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-in {
        animation: fadeInUp 0.6s ease-out;
    }

    /* 隐藏默认 footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 模型加载
# ============================================================
@st.cache_resource
def init_model():
    model, vectorizer = load_model()
    n2c = load_class_names()
    conf = Config()
    return model, vectorizer, n2c, conf


@st.cache_data
def load_dashboard_data(_conf):
    """加载看板数据"""
    train = pd.read_csv(_conf.train_datapath, sep='\t', names=['text', 'label'])
    test = pd.read_csv(_conf.test_datapath, sep='\t', names=['text', 'label'])
    dev = pd.read_csv(_conf.dev_datapath, sep='\t', names=['text', 'label'])
    n2c, _ = _conf.load_class_map()

    label_counts = Counter(train['label'])
    label_stats = []
    for lid, cnt in label_counts.most_common():
        label_stats.append({
            'label_id': lid,
            'label_name': n2c.get(lid, '?'),
            'count': cnt,
            'pct': cnt / len(train) * 100,
        })

    return train, test, dev, n2c, label_stats


# 初始化
model, vectorizer, n2c, conf = init_model()
train_df, test_df, dev_df, n2c_full, label_stats = load_dashboard_data(conf)


# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0;">
        <h2 style="color:white; margin:0;">🤖 AI Classifier</h2>
        <p style="color:rgba(255,255,255,0.5); font-size:0.85rem;">商品智能分类 v2.0</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # 模型状态
    st.markdown("""
    <div style="background:rgba(56,239,125,0.15); border:1px solid rgba(56,239,125,0.3);
                border-radius:12px; padding:16px; margin-bottom:16px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <div style="width:10px; height:10px; background:#38ef7d; border-radius:50%;
                        box-shadow:0 0 10px rgba(56,239,125,0.5);"></div>
            <span style="color:#38ef7d; font-weight:600;">模型运行中</span>
        </div>
        <div style="color:rgba(255,255,255,0.6); font-size:0.8rem; margin-top:8px;">
            算法: Random Forest<br>
            类别: {n_classes} 类<br>
            特征: TF-IDF (50k)
        </div>
    </div>
    """.format(n_classes=model.n_classes_), unsafe_allow_html=True)

    # 快速统计
    total_samples = len(train_df) + len(test_df) + len(dev_df)
    st.markdown(f"""
    <div style="background:rgba(255,255,255,0.05); border-radius:12px; padding:16px;">
        <p style="color:rgba(255,255,255,0.5); font-size:0.75rem; margin:0;">数据集规模</p>
        <p style="color:white; font-size:1.5rem; font-weight:700; margin:4px 0;">{total_samples:,}</p>
        <p style="color:rgba(255,255,255,0.4); font-size:0.75rem; margin:0;">
            训练 {len(train_df):,} / 测试 {len(test_df):,} / 验证 {len(dev_df):,}
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # 导航
    st.markdown("""
    <p style="color:rgba(255,255,255,0.4); font-size:0.7rem; text-transform:uppercase;
              letter-spacing:2px; margin-bottom:8px;">导航</p>
    """, unsafe_allow_html=True)


# ============================================================
# 主区域
# ============================================================
st.markdown("""
<div class="animate-in">
    <h1 style="font-size:2.5rem; font-weight:800; margin-bottom:0;">
        🤖 商品智能分类
    </h1>
    <p style="color:#888; font-size:1rem; margin-top:4px;">
        Random Forest + TF-IDF · 30 品类 · 实时推理
    </p>
</div>
""", unsafe_allow_html=True)

# ========== 四个标签页 ==========
tab1, tab2, tab3, tab4 = st.tabs([
    '🎯 智能预测',
    '📋 批量处理',
    '📊 数据看板',
    '🔬 模型洞察',
])

# ============================================================
# Tab 1: 智能预测
# ============================================================
with tab1:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### 输入商品名称")
        st.markdown('<div class="search-box">', unsafe_allow_html=True)

        # 快捷示例
        examples = ['伊利纯牛奶250ml', '小米14 Pro 手机壳', '三只松鼠每日坚果750g',
                     '茅台飞天53度500ml', '维达超韧抽纸3层', '乐高城市系列积木']
        example_cols = st.columns(3)
        selected_example = None
        for i, (col, ex) in enumerate(zip(example_cols, examples)):
            if col.button(ex, key=f'ex_{i}', use_container_width=True):
                selected_example = ex

        text = st.text_input(
            '商品名称',
            value=selected_example if selected_example else '',
            placeholder='输入商品名称，如: 伊利纯牛奶250ml...',
            label_visibility='collapsed',
        )
        st.markdown('</div>', unsafe_allow_html=True)

        predict_btn = st.button('🔍 开始识别', type='primary', use_container_width=True)

    with col_right:
        if text or predict_btn:
            text_to_predict = text.strip()

            if text_to_predict:
                with st.spinner('🧠 AI 推理中...'):
                    start = time.time()
                    pred_id, pred_name = predict_single(text_to_predict)
                    _, _, proba = predict_with_proba([text_to_predict])
                    elapsed = time.time() - start

                top_proba = proba[0][pred_id]

                # 置信度等级
                if top_proba >= 0.5:
                    conf_level, conf_color, conf_emoji = '高', '#38ef7d', '🟢'
                elif top_proba >= 0.2:
                    conf_level, conf_color, conf_emoji = '中', '#ffc107', '🟡'
                else:
                    conf_level, conf_color, conf_emoji = '低', '#f5576c', '🔴'

                # 主预测卡片
                st.markdown(f"""
                <div class="predict-card animate-in">
                    <p style="opacity:0.6; font-size:0.85rem; margin:0;">预测结果</p>
                    <p class="pred-label" style="margin:8px 0;">{pred_name}</p>
                    <div style="display:flex; align-items:center; gap:12px; margin:12px 0;">
                        <div style="flex:1; background:rgba(255,255,255,0.1); border-radius:6px; height:8px;">
                            <div class="conf-bar" style="width:{top_proba*100}%; background:{conf_color};
                                        border-radius:6px;"></div>
                        </div>
                        <span style="font-weight:700; font-size:1.1rem;">{top_proba:.1%}</span>
                        <span style="font-size:1.5rem;">{conf_emoji}</span>
                    </div>
                    <p style="opacity:0.4; font-size:0.75rem;">置信度: {conf_level} · 耗时 {elapsed*1000:.1f}ms</p>
                </div>
                """, unsafe_allow_html=True)

                # Top5 概率
                st.markdown("#### 🏆 Top 5 预测概率")

                top5_idx = proba[0].argsort()[::-1][:5]
                fig = go.Figure()

                colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe']
                names = [n2c.get(i, str(i)) for i in top5_idx]
                values = [proba[0][i] for i in top5_idx]

                fig.add_trace(go.Bar(
                    y=names[::-1],
                    x=values[::-1],
                    orientation='h',
                    marker=dict(
                        color=colors[::-1],
                        line=dict(width=0),
                    ),
                    text=[f'{v:.1%}' for v in values[::-1]],
                    textposition='outside',
                    textfont=dict(color='white', size=14),
                ))

                fig.update_layout(
                    height=250,
                    margin=dict(l=0, r=80, t=10, b=10),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(
                        showgrid=False,
                        showticklabels=False,
                        range=[0, max(values) * 1.2],
                    ),
                    yaxis=dict(
                        showgrid=False,
                        tickfont=dict(size=13, color='#ccc'),
                    ),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # 下方：最近预测历史
    if 'history' not in st.session_state:
        st.session_state.history = []

    if text and predict_btn and text.strip():
        st.session_state.history.insert(0, {
            'time': datetime.now().strftime('%H:%M:%S'),
            'text': text.strip(),
            'pred': pred_name,
            'conf': f'{top_proba:.1%}',
        })
        st.session_state.history = st.session_state.history[:20]  # 最多20条

    if st.session_state.history:
        st.markdown("---")
        st.markdown("#### 📜 预测历史")
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(
            hist_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'time': st.column_config.TextColumn('时间', width='small'),
                'text': st.column_config.TextColumn('商品名称', width='large'),
                'pred': st.column_config.TextColumn('预测类别', width='medium'),
                'conf': st.column_config.TextColumn('置信度', width='small'),
            },
        )


# ============================================================
# Tab 2: 批量处理
# ============================================================
with tab2:
    st.markdown("### 📋 批量预测")

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded_file = st.file_uploader(
            '拖拽或点击上传文件',
            type=['txt', 'tsv', 'csv'],
            help='支持 TSV (text\\tlabel) 或单列文本文件',
        )
    with col2:
        st.markdown("""
        <div style="background:rgba(102,126,234,0.1); border:1px solid rgba(102,126,234,0.2);
                    border-radius:12px; padding:20px; height:100%;">
            <p style="font-weight:600; margin:0 0 8px 0;">📌 支持格式</p>
            <p style="font-size:0.85rem; color:#888; margin:0;">
                <b>TSV:</b> 商品名<code>\\t</code>标签<br>
                <b>TXT:</b> 每行一个商品名<br>
                <b>CSV:</b> 含 text 列
            </p>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode('utf-8')
            lines = [l.strip() for l in content.split('\n') if l.strip()]

            if '\t' in lines[0]:
                parts = [l.split('\t') for l in lines]
                df = pd.DataFrame(parts, columns=['text', 'label'])
                has_label = True
            else:
                df = pd.DataFrame({'text': lines})
                has_label = False

            st.markdown(f"""
            <div style="background:rgba(79,172,254,0.1); border:1px solid rgba(79,172,254,0.2);
                        border-radius:12px; padding:16px; margin:12px 0;">
                <span style="font-weight:600;">📁 {uploaded_file.name}</span>
                &nbsp;·&nbsp; {len(df):,} 条数据
                &nbsp;·&nbsp; {'含标注' if has_label else '纯文本'}
            </div>
            """, unsafe_allow_html=True)

            with st.expander("👁 预览数据", expanded=False):
                st.dataframe(df.head(20), use_container_width=True)

            if st.button('🚀 开始批量预测', type='primary', use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                texts = df['text'].tolist()
                batch_size = 500

                all_pred_ids = []
                all_pred_names = []

                start = time.time()
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    pids, pnames = predict_batch(batch)
                    all_pred_ids.extend(pids)
                    all_pred_names.extend(pnames)
                    progress = (i + len(batch)) / len(texts)
                    progress_bar.progress(progress)
                    status_text.text(f'处理中... {min(i+batch_size, len(texts)):,}/{len(texts):,}')

                elapsed = time.time() - start
                progress_bar.empty()
                status_text.empty()

                df['pred_label_id'] = all_pred_ids
                df['pred_label_name'] = all_pred_names

                # 结果统计
                st.markdown("---")
                result_cols = st.columns(4)
                with result_cols[0]:
                    st.metric('处理总量', f'{len(df):,} 条')
                with result_cols[1]:
                    st.metric('耗时', f'{elapsed:.2f}s')
                with result_cols[2]:
                    st.metric('速度', f'{len(df)/elapsed:.0f} 条/秒')
                with result_cols[3]:
                    if has_label:
                        acc = (df['label'].astype(int) == df['pred_label_id']).mean()
                        st.metric('准确率', f'{acc:.2%}')

                # 分布图
                if has_label:
                    st.markdown("#### 预测分布 vs 真实分布")
                    fig_cols = st.columns(2)
                    with fig_cols[0]:
                        dist_true = df['label'].value_counts().head(10)
                        fig1 = px.bar(
                            x=dist_true.index, y=dist_true.values,
                            title='真实标签 Top 10',
                            labels={'x': '', 'y': '数量'},
                            color_discrete_sequence=['#667eea'],
                        )
                        st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
                    with fig_cols[1]:
                        dist_pred = df['pred_label_id'].value_counts().head(10)
                        fig2 = px.bar(
                            x=dist_pred.index, y=dist_pred.values,
                            title='预测标签 Top 10',
                            labels={'x': '', 'y': '数量'},
                            color_discrete_sequence=['#f5576c'],
                        )
                        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

                # 结果表格
                st.markdown("#### 预测结果")
                st.dataframe(df, use_container_width=True, hide_index=True)

                # 下载
                csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    '📥 下载完整结果 (CSV)',
                    data=csv,
                    file_name=f'predict_result_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv',
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f'处理失败: {e}')


# ============================================================
# Tab 3: 数据看板
# ============================================================
with tab3:
    st.markdown("### 📊 数据看板")

    # 顶行指标卡
    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">训练样本</div>
            <div class="value">{len(train_df):,}</div>
            <div class="sub">80.0% of total</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[1]:
        st.markdown(f"""
        <div class="metric-card green">
            <div class="label">类别数量</div>
            <div class="value">{len(n2c_full)}</div>
            <div class="sub">商品品类</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[2]:
        max_label = label_stats[0]
        st.markdown(f"""
        <div class="metric-card orange">
            <div class="label">最大类别</div>
            <div class="value">{max_label['label_name']}</div>
            <div class="sub">{max_label['count']:,} 条 ({max_label['pct']:.1f}%)</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_cols[3]:
        imbalance_ratio = label_stats[0]['count'] / label_stats[-1]['count']
        st.markdown(f"""
        <div class="metric-card blue">
            <div class="label">不平衡比</div>
            <div class="value">{imbalance_ratio:.0f}:1</div>
            <div class="sub">max / min</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 图表行
    chart_cols = st.columns(2)

    with chart_cols[0]:
        st.markdown("#### 类别样本分布")
        chart_data = pd.DataFrame(label_stats)
        fig = px.bar(
            chart_data,
            x='label_name', y='count',
            color='count',
            color_continuous_scale='Viridis',
            labels={'label_name': '类别', 'count': '样本数'},
            height=450,
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            margin=dict(l=10, r=10, t=30, b=80),
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with chart_cols[1]:
        st.markdown("#### 数据划分占比")
        pie_fig = go.Figure(data=[go.Pie(
            labels=['训练集', '测试集', '验证集'],
            values=[len(train_df), len(test_df), len(dev_df)],
            hole=0.45,
            marker=dict(colors=['#667eea', '#f5576c', '#38ef7d']),
            textinfo='label+percent',
            textfont=dict(size=14),
        )])
        pie_fig.update_layout(
            height=450,
            margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(pie_fig, use_container_width=True, config={'displayModeBar': False})

    # 类别表格
    with st.expander("🔍 全部类别详情", expanded=False):
        search_term = st.text_input('搜索类别', placeholder='输入类别名筛选...', key='class_search')
        classes_df = pd.DataFrame(label_stats)
        classes_df['占比'] = classes_df['pct'].apply(lambda x: f'{x:.2f}%')
        classes_df = classes_df.rename(columns={
            'label_id': 'ID', 'label_name': '类别名称',
            'count': '样本数',
        })
        display_df = classes_df[['ID', '类别名称', '样本数', '占比']]
        if search_term:
            display_df = display_df[display_df['类别名称'].str.contains(search_term)]
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ============================================================
# Tab 4: 模型洞察
# ============================================================
with tab4:
    st.markdown("### 🔬 模型洞察")

    model_cols = st.columns(3)
    with model_cols[0]:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #434343 0%, #000000 100%);">
            <div class="label">算法</div>
            <div class="value" style="font-size:1.5rem;">Random Forest</div>
            <div class="sub">n_estimators={model.n_estimators}</div>
        </div>
        """, unsafe_allow_html=True)

    with model_cols[1]:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #434343 0%, #000000 100%);">
            <div class="label">特征工程</div>
            <div class="value" style="font-size:1.5rem;">TF-IDF</div>
            <div class="sub">50,000 features, ngram(1,2)</div>
        </div>
        """, unsafe_allow_html=True)

    with model_cols[2]:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #434343 0%, #000000 100%);">
            <div class="label">最大深度</div>
            <div class="value" style="font-size:1.5rem;">{model.max_depth}</div>
            <div class="sub">min_samples_leaf={model.min_samples_leaf}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 模型参数表
    param_col1, param_col2 = st.columns(2)
    with param_col1:
        st.markdown("#### ⚙️ 超参数")
        params_df = pd.DataFrame([
            {'参数': 'n_estimators', '值': model.n_estimators, '说明': '决策树数量'},
            {'参数': 'max_depth', '值': model.max_depth, '说明': '最大深度'},
            {'参数': 'min_samples_split', '值': model.min_samples_split, '说明': '分裂最小样本'},
            {'参数': 'min_samples_leaf', '值': model.min_samples_leaf, '说明': '叶节点最小样本'},
            {'参数': 'max_features', '值': model.max_features, '说明': '特征采样策略'},
            {'参数': 'class_weight', '值': model.class_weight, '说明': '类别权重'},
        ])
        st.dataframe(params_df, use_container_width=True, hide_index=True)

    with param_col2:
        st.markdown("#### 📈 性能基准")
        metrics_df = pd.DataFrame([
            {'指标': 'Dev Accuracy', '值': '45.34%'},
            {'指标': 'Test Accuracy', '值': '46.23%'},
            {'指标': 'Macro F1 (Dev)', '值': '47.35%'},
            {'指标': 'Macro F1 (Test)', '值': '49.19%'},
            {'指标': '训练耗时', '值': '~8s'},
            {'指标': '推理速度', '值': '~0.3ms/条'},
            {'指标': '模型大小', '值': '22.2 MB'},
        ])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 快速测试
    st.markdown("#### 🧪 实时推理测试")
    test_text = st.text_input('输入测试文本', placeholder='输入任意商品名称...', key='insight_test')
    if test_text:
        col_a, col_b = st.columns([2, 1])
        with col_a:
            pred_id, pred_name = predict_single(test_text)
            _, _, proba = predict_with_proba([test_text])
            st.success(f'预测: **{pred_name}** (ID: {pred_id}, 置信度: {proba[0][pred_id]:.1%})')

            # 所有类别概率
            all_probs = [(n2c.get(i, str(i)), proba[0][i]) for i in range(len(n2c))]
            all_probs.sort(key=lambda x: -x[1])
            prob_fig = px.bar(
                x=[p[0] for p in all_probs],
                y=[p[1] for p in all_probs],
                title='完整概率分布',
                labels={'x': '', 'y': '概率'},
                color_discrete_sequence=['#667eea'],
                height=300,
            )
            prob_fig.update_layout(xaxis_tickangle=-45, margin=dict(l=10, r=10, t=30, b=80))
            st.plotly_chart(prob_fig, use_container_width=True, config={'displayModeBar': False})

        with col_b:
            st.markdown("##### 🔥 Top 5")
            top5 = all_probs[:5]
            for name, prob in top5:
                color = '#38ef7d' if prob > 0.3 else '#ffc107' if prob > 0.1 else '#888'
                st.markdown(f"""
                <div style="display:flex; align-items:center; justify-content:space-between;
                            padding:8px 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                    <span style="font-weight:500;">{name}</span>
                    <span style="color:{color}; font-weight:600;">{prob:.1%}</span>
                </div>
                """, unsafe_allow_html=True)


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown(f"""
<div style="text-align:center; color:#666; font-size:0.8rem; padding:20px 0;">
    🤖 AI 商品分类系统 v2.0 · Random Forest + TF-IDF
    &nbsp;·&nbsp; 最后训练: {datetime.now().strftime('%Y-%m-%d')}
</div>
""", unsafe_allow_html=True)
