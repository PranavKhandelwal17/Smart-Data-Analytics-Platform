import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import google.generativeai as genai
import numpy as np
import logging
import traceback
from datetime import datetime

# ===========================
# PAGE CONFIG
# ===========================
st.set_page_config(
    page_title="Smart Data Analytics Platform",
    page_icon="📊",
    layout="wide"
)

# ===========================
# SMART ERROR LOGGING
# ===========================
class AppLogger:
    def __init__(self):
        self.logger = logging.getLogger("analytics_app")
        self.logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler("app_errors.log")
        file_handler.setLevel(logging.ERROR)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
    
    def log_error(self, error_msg, exception=None):
        """Log error with full traceback"""
        if exception:
            self.logger.error(f"{error_msg}: {str(exception)}\n{traceback.format_exc()}")
        else:
            self.logger.error(error_msg)
    
    def log_info(self, msg):
        self.logger.info(msg)

logger = AppLogger()

# ===========================
# SESSION STATE
# ===========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "filters_applied" not in st.session_state:
    st.session_state.filters_applied = False

if "original_df" not in st.session_state:
    st.session_state.original_df = None

if "column_roles" not in st.session_state:
    st.session_state.column_roles = {}

# ===========================
# GEMINI SETUP
# ===========================
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel("models/gemini-flash-lite-latest")
except Exception as e:
    logger.log_error("Failed to configure Gemini", e)
    model = None

# ===========================
# COLUMN ROLE DETECTION
# ===========================
@st.cache_data(ttl=3600)
def detect_column_roles(df):
    """
    Detect column roles: ID, Measure (numeric), Dimension (categorical)
    Returns a dictionary with column names and their roles
    """
    roles = {}
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Check for ID columns
        id_patterns = ['id', 'key', 'code', 'number', 'no.', 'num', '_id']
        if any(pattern in col_lower for pattern in id_patterns):
            # Check if unique values equal row count (high cardinality)
            if df[col].nunique() == len(df) or df[col].nunique() > len(df) * 0.5:
                roles[col] = "ID"
                continue
        
        # Check numeric columns - likely measures
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() > 1:  # Not a constant
                roles[col] = "Measure"
            else:
                roles[col] = "Constant"
            continue
        
        # Check for date/time columns
        date_patterns = ['date', 'time', 'datetime', 'created', 'updated', 'timestamp']
        if any(pattern in col_lower for pattern in date_patterns):
            roles[col] = "Date/Time"
            continue
        
        # Check categorical/string columns - likely dimensions
        unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
        
        if unique_ratio < 0.5 and df[col].nunique() <= 50:
            roles[col] = "Dimension"
        elif df[col].nunique() <= 20:
            roles[col] = "Dimension"
        else:
            roles[col] = "Text"
    
    return roles


def display_column_roles(roles):
    """Display column roles in a formatted way"""
    id_cols = [k for k, v in roles.items() if v == "ID"]
    measure_cols = [k for k, v in roles.items() if v == "Measure"]
    dimension_cols = [k for k, v in roles.items() if v == "Dimension"]
    date_cols = [k for k, v in roles.items() if v == "Date/Time"]
    text_cols = [k for k, v in roles.items() if v == "Text"]
    
    st.markdown("### 🔍 Column Role Detection")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**🆔 IDs**")
        if id_cols:
            for c in id_cols:
                st.code(c, language=None)
        else:
            st.caption("None detected")
    
    with col2:
        st.markdown("**📊 Measures**")
        if measure_cols:
            for c in measure_cols[:5]:
                st.code(c, language=None)
            if len(measure_cols) > 5:
                st.caption(f"+{len(measure_cols)-5} more")
        else:
            st.caption("None detected")
    
    with col3:
        st.markdown("**🏷️ Dimensions**")
        if dimension_cols:
            for c in dimension_cols[:5]:
                st.code(c, language=None)
            if len(dimension_cols) > 5:
                st.caption(f"+{len(dimension_cols)-5} more")
        else:
            st.caption("None detected")
    
    with col4:
        st.markdown("**📅 Date/Time**")
        if date_cols:
            for c in date_cols:
                st.code(c, language=None)
        else:
            st.caption("None detected")


# ===========================
# SMART COLUMN FINDER
# ===========================
def find_matching_column(kpi, df, col_type='numeric'):
    """
    Find the best matching column based on KPI text.
    Uses fuzzy/partial matching to find columns.
    """
    if col_type == 'numeric':
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if not cols:
        return None
    
    kpi_lower = kpi.lower()
    
    # First, try exact match
    for col in cols:
        if col.lower() == kpi_lower:
            return col
    
    # Try partial match (contains)
    for col in cols:
        col_lower = col.lower()
        kpi_words = kpi_lower.replace('_', ' ').replace('-', ' ').split()
        for word in kpi_words:
            if len(word) > 2 and word in col_lower:
                return col
    
    # Try reverse - column name in KPI
    for col in cols:
        col_lower = col.lower()
        for i in range(len(col_lower)):
            if col_lower[i:] in kpi_lower:
                return col
    
    # Common keyword mappings
    keyword_map = {
        'sales': ['sales', 'revenue', 'income', 'turnover'],
        'profit': ['profit', 'gain', 'earning'],
        'quantity': ['qty', 'quantity', 'units', 'count', 'volume'],
        'price': ['price', 'cost', 'amount', 'value'],
        'rating': ['rating', 'score', 'review'],
        'discount': ['discount', 'off', 'reduction'],
    }
    
    for key, keywords in keyword_map.items():
        if key in kpi_lower:
            for col in cols:
                col_lower = col.lower()
                for kw in keywords:
                    if kw in col_lower:
                        return col
    
    return cols[0] if cols else None


# ===========================
# AI-POWERED KPI COLUMN SELECTION
# ===========================
@st.cache_data(ttl=1800)
def get_ai_column_suggestions(df, kpi_description):
    """
    AI-powered column selection for KPIs
    """
    if not model:
        return None
    
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        prompt = f"""
        Given a dataset with these columns:
        Numeric: {numeric_cols}
        Categorical: {categorical_cols}
        
        For this KPI request: "{kpi_description}"
        
        Suggest the best numeric column to use.
        Return ONLY the column name, nothing else.
        """
        
        response = model.generate_content(prompt)
        suggested_col = response.text.strip()
        
        # Verify the column exists
        for col in numeric_cols:
            if col.lower() == suggested_col.lower():
                return col
        
        return None
    except Exception as e:
        logger.log_error("AI column suggestion failed", e)
        return None


# ===========================
# EXECUTIVE SUMMARY GENERATION
# ===========================
@st.cache_data(ttl=3600)
def generate_executive_summary(df, roles):
    """
    Generate AI-powered executive summary
    """
    if not model:
        return "Configure Gemini API for AI summaries"
    
    try:
        measure_cols = [k for k, v in roles.items() if v == "Measure"]
        dimension_cols = [k for k, v in roles.items() if v == "Dimension"]
        
        # Get basic stats
        stats = df[measure_cols[:3]].describe().to_string() if measure_cols else "No numeric data"
        
        prompt = f"""
        Generate a brief executive summary (3-5 bullet points) for this dataset.
        
        Dataset Shape: {df.shape}
        Key Measures: {measure_cols[:5]}
        Key Dimensions: {dimension_cols[:5]}
        
        Statistical Summary:
        {stats}
        
        Focus on:
        - Key insights from the data
        - Important trends or patterns
        - Business recommendations
        
        Keep it concise and professional.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.log_error("Executive summary generation failed", e)
        return "Unable to generate summary"


# ===========================
# UNIVERSAL KPI ENGINE
# ===========================
@st.cache_data(ttl=600)
def generate_universal_kpi(kpi_text, df_json, date_col=None):
    """
    Universal KPI Engine that works with ANY data type and column names.
    Cached for performance.
    """
    # Convert JSON back to DataFrame (needed for caching)
    from io import StringIO
    df = pd.read_json(StringIO(df_json), orient='split')
    
    if df.empty:
        return "N/A", "N/A"
    
    kpi = kpi_text.lower().strip()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return "No numeric data", "N/A"
    
    # Find the best matching column for this KPI
    matched_col = find_matching_column(kpi, df, 'numeric')
    matched_cat_col = find_matching_column(kpi, df, 'categorical')
    
    # ---------------------------
    # CATEGORY WISE / BY CATEGORY
    # ---------------------------
    if ('category' in kpi or 'by category' in kpi or 'group by' in kpi) and matched_col:
        if categorical_cols:
            cat_col = matched_cat_col if matched_cat_col else categorical_cols[0]
            grouped = df.groupby(cat_col)[matched_col].sum().sort_values(ascending=False)
            top_5 = grouped.head(5).to_dict()
            return top_5, f"{matched_col} by {cat_col}"
    
    # ---------------------------
    # DISTRIBUTION
    # ---------------------------
    if 'distribution' in kpi and matched_col:
        dist_data = {
            'mean': round(df[matched_col].mean(), 2),
            'std': round(df[matched_col].std(), 2),
            'min': round(df[matched_col].min(), 2),
            '25%': round(df[matched_col].quantile(0.25), 2),
            '50%': round(df[matched_col].quantile(0.50), 2),
            '75%': round(df[matched_col].quantile(0.75), 2),
            'max': round(df[matched_col].max(), 2),
        }
        return dist_data, f"Distribution of {matched_col}"
    
    # ---------------------------
    # TREND
    # ---------------------------
    if 'trend' in kpi and date_col and matched_col:
        try:
            df_sorted = df.sort_values(date_col)
            trend_data = df_sorted.groupby(date_col)[matched_col].sum().to_dict()
            return trend_data, f"Trend of {matched_col}"
        except:
            pass
    
    # ---------------------------
    # COUNT / TOTAL RECORDS
    # ---------------------------
    if any(x in kpi for x in ['total record', 'total count', 'total row', 'number of', 'count of', 'how many']):
        return len(df), "Records"
    
    # ---------------------------
    # UNIQUE / DISTINCT COUNT
    # ---------------------------
    if 'unique' in kpi or 'distinct' in kpi:
        for col in categorical_cols:
            if any(x in kpi for x in [col.lower(), 'category', 'product', 'customer', 'id']):
                return df[col].nunique(), f"Unique {col}"
        for col in numeric_cols:
            if any(x in kpi for x in [col.lower()]):
                return df[col].nunique(), f"Unique {col}"
        return df.nunique().sum(), "Total Unique Values"
    
    # ---------------------------
    # SUM / TOTAL
    # ---------------------------
    if 'total' in kpi or 'sum' in kpi:
        if matched_col:
            return round(df[matched_col].sum(), 2), f"Total {matched_col}"
        return round(df[numeric_cols[0]].sum(), 2), f"Total {numeric_cols[0]}"
    
    # ---------------------------
    # AVERAGE / MEAN
    # ---------------------------
    if 'average' in kpi or 'mean' in kpi or 'avg' in kpi:
        if matched_col:
            return round(df[matched_col].mean(), 2), f"Avg {matched_col}"
        return round(df[numeric_cols[0]].mean(), 2), f"Avg {numeric_cols[0]}"
    
    # ---------------------------
    # MEDIAN
    # ---------------------------
    if 'median' in kpi:
        if matched_col:
            return round(df[matched_col].median(), 2), f"Median {matched_col}"
        return round(df[numeric_cols[0]].median(), 2), f"Median {numeric_cols[0]}"
    
    # ---------------------------
    # MAX / HIGHEST / TOP
    # ---------------------------
    if 'max' in kpi or 'highest' in kpi or 'largest' in kpi:
        if matched_col:
            return round(df[matched_col].max(), 2), f"Max {matched_col}"
        return round(df[numeric_cols[0]].max(), 2), f"Max {numeric_cols[0]}"
    
    # ---------------------------
    # MIN / LOWEST / BOTTOM
    # ---------------------------
    if 'min' in kpi or 'lowest' in kpi or 'smallest' in kpi or 'bottom' in kpi:
        if matched_col:
            return round(df[matched_col].min(), 2), f"Min {matched_col}"
        return round(df[numeric_cols[0]].min(), 2), f"Min {numeric_cols[0]}"
    
    # ---------------------------
    # STANDARD DEVIATION
    # ---------------------------
    if 'std' in kpi or 'standard deviation' in kpi or 'variation' in kpi:
        if matched_col:
            return round(df[matched_col].std(), 2), f"Std {matched_col}"
        return round(df[numeric_cols[0]].std(), 2), f"Std {numeric_cols[0]}"
    
    # ---------------------------
    # PERCENTILE
    # ---------------------------
    percentiles = {'25th': 0.25, '50th': 0.5, '75th': 0.75, '90th': 0.90, '95th': 0.95, '99th': 0.99}
    for pct_name, pct_val in percentiles.items():
        if pct_name in kpi:
            if matched_col:
                return round(df[matched_col].quantile(pct_val), 2), f"{pct_name} {matched_col}"
            return round(df[numeric_cols[0]].quantile(pct_val), 2), f"{pct_name} {numeric_cols[0]}"
    
    # ---------------------------
    # GROWTH RATE (Date-based)
    # ---------------------------
    if 'growth' in kpi or 'change' in kpi:
        if date_col and numeric_cols:
            try:
                df_sorted = df.sort_values(date_col)
                col = matched_col if matched_col else numeric_cols[0]
                first_val = df_sorted[col].iloc[0]
                last_val = df_sorted[col].iloc[-1]
                if first_val != 0:
                    growth = ((last_val - first_val) / first_val) * 100
                    return round(growth, 2), f"Growth %"
            except:
                pass
    
    # ---------------------------
    # CATEGORICAL - TOP VALUE
    # ---------------------------
    if 'top' in kpi or 'most common' in kpi or 'popular' in kpi:
        cat_col = find_matching_column(kpi, df, 'categorical')
        if cat_col:
            top_val = df[cat_col].mode()
            if len(top_val) > 0:
                return str(top_val[0]), f"Top {cat_col}"
        if categorical_cols:
            top_val = df[categorical_cols[0]].mode()
            if len(top_val) > 0:
                return str(top_val[0]), f"Top {categorical_cols[0]}"
    
    # ---------------------------
    # DEFAULT - Show stats for matched column
    # ---------------------------
    if matched_col:
        return round(df[matched_col].sum(), 2), f"{matched_col}"
    
    return "N/A", "No Data"


# ===========================
# AUTO KPI GENERATOR (CACHED)
# ===========================
@st.cache_data(ttl=600)
def generate_auto_kpis(df_json, date_col=None):
    """
    Automatically generate KPIs based on the data available.
    Cached for performance.
    """
    from io import StringIO
    df = pd.read_json(StringIO(df_json), orient='split')
    
    kpis = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Basic count KPI
    kpis.append(("Total Records", len(df), "Records"))
    
    # Numeric KPIs - generate for all numeric columns
    for col in numeric_cols[:5]:
        kpis.append((f"Total {col}", round(df[col].sum(), 2), f"Sum"))
        kpis.append((f"Average {col}", round(df[col].mean(), 2), f"Mean"))
        kpis.append((f"Min {col}", round(df[col].min(), 2), f"Min"))
        kpis.append((f"Max {col}", round(df[col].max(), 2), f"Max"))
    
    # Categorical KPIs
    for col in categorical_cols[:2]:
        top_val = df[col].mode()
        if len(top_val) > 0:
            kpis.append((f"Top {col}", str(top_val[0]), "Category"))
        kpis.append((f"Unique {col}", df[col].nunique(), "Unique Count"))
    
    # Date-based KPIs
    if date_col:
        try:
            df_sorted = df.sort_values(date_col)
            if numeric_cols:
                first_val = df_sorted[numeric_cols[0]].iloc[0]
                last_val = df_sorted[numeric_cols[0]].iloc[-1]
                if first_val > 0:
                    growth = ((last_val - first_val) / first_val) * 100
                    kpis.append(("Growth Rate", round(growth, 2), "%"))
        except:
            pass
    
    return kpis


# ===========================
# OUTLIER REMOVAL (IQR)
# ===========================
def remove_outliers_iqr(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df


# ===========================
# DATASET PROFILE REPORT
# ===========================
def generate_dataset_profile(df, roles):
    """Generate a pandas-profiling style dataset profile"""
    
    st.markdown("---")
    st.markdown("## 📋 Dataset Profile Report")
    
    # Overview Section
    with st.expander("📊 Overview", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", df.shape[0])
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        col4.metric("Duplicate Rows", df.duplicated().sum())
    
    # Data Types Section
    with st.expander("🔢 Data Types"):
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Data Type': [str(df[col].dtype) for col in df.columns],
            'Role': [roles.get(col, 'Unknown') for col in df.columns],
            'Unique Values': [df[col].nunique() for col in df.columns],
            'Missing Values': [df[col].isnull().sum() for col in df.columns],
            'Missing %': [round(df[col].isnull().sum() / len(df) * 100, 2) for col in df.columns]
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    # Statistical Summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        with st.expander("📈 Statistical Summary"):
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    # Missing Values Analysis
    with st.expander("❌ Missing Values Analysis"):
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Column': missing[missing > 0].index,
            'Missing Count': missing[missing > 0].values,
            'Missing %': missing_pct[missing > 0].values
        })
        if not missing_df.empty:
            st.dataframe(missing_df, use_container_width=True)
            fig = px.bar(missing_df, x='Column', y='Missing %', 
                        title="Missing Values Percentage",
                        color='Missing %',
                        color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values in the dataset!")
    
    # Distribution Analysis
    with st.expander("📊 Distribution Analysis"):
        if numeric_cols:
            selected_col = st.selectbox("Select Column for Distribution", numeric_cols)
            col1, col2 = st.columns(2)
            
            with col1:
                fig_hist = px.histogram(df, x=selected_col, 
                                       title=f"Histogram of {selected_col}",
                                       nbins=30)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Box plot
                fig_box = px.box(df, y=selected_col,
                               title=f"Box Plot of {selected_col}")
                st.plotly_chart(fig_box, use_container_width=True)
    
    # Correlation Matrix
    with st.expander("🔗 Correlation Matrix"):
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                          title="Correlation Heatmap",
                          color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation analysis")
    
    # Sample Data
    with st.expander("👀 Sample Data"):
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Showing first 10 of {len(df)} rows")


# ===========================
# ENHANCED DASHBOARD GENERATOR
# ===========================
def generate_enhanced_dashboard(df, date_col=None):
    """
    Generate comprehensive dashboard with multiple visualizations.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    tabs = st.tabs(["📊 Overview", "📈 Trends", "📉 Distributions", "🔗 Correlations", "🏆 Top/Bottom"])
    
    # Tab 1: Overview with key metrics
    with tabs[0]:
        if numeric_cols:
            col1, col2, col3, col4 = st.columns(4)
            
            for i, col in enumerate(numeric_cols[:4]):
                with [col1, col2, col3, col4][i]:
                    st.metric(f"Avg {col}", round(df[col].mean(), 2))
            
            st.markdown("### 📋 Summary Statistics")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
    
    # Tab 2: Time series trends
    with tabs[1]:
        if date_col and numeric_cols:
            selected_col = st.selectbox("Select Metric for Trend", numeric_cols, key="trend_select")
            trend_df = df.groupby(date_col)[selected_col].sum().reset_index()
            fig = px.line(trend_df, x=date_col, y=selected_col, 
                         title=f"{selected_col} Over Time",
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Date column not detected for trend analysis.")
    
    # Tab 3: Distributions
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            if numeric_cols:
                selected_col = st.selectbox("Select Column for Histogram", numeric_cols, key="hist")
                fig = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if categorical_cols:
                selected_cat = st.selectbox("Select Category for Pie Chart", categorical_cols, key="pie")
                top_cats = df[selected_cat].value_counts().head(10)
                fig = px.pie(values=top_cats.values, names=top_cats.index, 
                            title=f"Top 10 {selected_cat}")
                st.plotly_chart(fig, use_container_width=True)
    
    # Tab 4: Correlations
    with tabs[3]:
        if len(numeric_cols) >= 2:
            selected_corr = st.multiselect("Select Columns for Correlation", 
                                          numeric_cols, 
                                          default=numeric_cols[:min(5, len(numeric_cols))])
            if len(selected_corr) >= 2:
                corr_matrix = df[selected_corr].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                              title="Correlation Heatmap")
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X Axis", selected_corr, key="scatter_x")
                with col2:
                    y_col = st.selectbox("Y Axis", selected_corr, index=1 if len(selected_corr) > 1 else 0, key="scatter_y")
                
                color_col = categorical_cols[0] if categorical_cols else None
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                               title=f"{x_col} vs {y_col}")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation analysis.")
    
    # Tab 5: Top/Bottom Analysis
    with tabs[4]:
        col1, col2 = st.columns(2)
        with col1:
            if numeric_cols:
                top_n = st.slider("Top N", 5, 20, 10)
                selected_col = st.selectbox("Column for Ranking", numeric_cols, key="rank")
                
                top_df = df.nlargest(top_n, selected_col)
                if categorical_cols:
                    fig = px.bar(top_df, x=categorical_cols[0], y=selected_col,
                               title=f"Top {top_n} {selected_col} by {categorical_cols[0]}")
                else:
                    fig = px.bar(top_df.reset_index(), x="index", y=selected_col,
                               title=f"Top {top_n} {selected_col}")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if numeric_cols:
                bottom_df = df.nsmallest(top_n, selected_col)
                if categorical_cols:
                    fig = px.bar(bottom_df, x=categorical_cols[0], y=selected_col,
                               title=f"Bottom {top_n} {selected_col} by {categorical_cols[0]}",
                               color_discrete_sequence=['red'])
                else:
                    fig = px.bar(bottom_df.reset_index(), x="index", y=selected_col,
                               title=f"Bottom {top_n} {selected_col}",
                               color_discrete_sequence=['red'])
                st.plotly_chart(fig, use_container_width=True)


# ===========================
# HEADER
# ===========================
st.markdown("""
<h1 style='text-align:center;'>📊 Smart Data Analytics Platform</h1>
<p style='text-align:center;'>Universal KPI Engine • Auto Dashboard • Advanced Cleaning • AI Chatbot</p>
<hr>
""", unsafe_allow_html=True)


# ===========================
# SIDEBAR
# ===========================
st.sidebar.header("⚙️ Controls")

# Error Display Section in Sidebar
with st.sidebar.expander("⚠️ Error Logs", expanded=False):
    try:
        with open("app_errors.log", "r") as f:
            errors = f.readlines()
            if errors:
                st.text("Recent errors:")
                for err in errors[-5:]:
                    st.text(err.strip()[:100])
            else:
                st.caption("No errors logged")
    except:
        st.caption("No errors logged")

st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
kpi_input = st.sidebar.text_input("Enter Custom KPIs (comma separated)")
auto_kpis = st.sidebar.checkbox("Generate Auto KPIs", value=True)

# AI Column Selection
st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI Column Selection")

# Only show if file is uploaded
if uploaded_file:
    if 'df_for_ai' not in st.session_state:
        st.session_state.df_for_ai = None
    
    ai_kpi_input = st.sidebar.text_input("AI Suggest Column (e.g., 'sales')")
    if ai_kpi_input and model:
        if st.sidebar.button("Get AI Suggestion"):
            try:
                from io import StringIO
                df_temp = pd.read_csv(uploaded_file)
                suggestion = get_ai_column_suggestions(df_temp, ai_kpi_input)
                if suggestion:
                    st.session_state.ai_suggestion = suggestion
                    st.sidebar.success(f"AI suggests: {suggestion}")
                else:
                    st.sidebar.warning("Could not determine column")
            except Exception as e:
                logger.log_error("AI suggestion failed", e)
                st.sidebar.error(f"Error: {str(e)[:50]}")

st.sidebar.markdown("---")
st.sidebar.subheader("🧹 Advanced Cleaning Controls")

remove_duplicates = st.sidebar.checkbox("Remove Duplicates")
remove_outliers = st.sidebar.checkbox("Remove Outliers (IQR)")
trim_text = st.sidebar.checkbox("Trim & Standardize Text")
convert_numeric = st.sidebar.checkbox("Auto Convert Numeric Strings")

handle_missing = st.sidebar.selectbox(
    "Handle Missing Values",
    ["None", "Drop Rows", "Fill with Mean", "Fill with Median", "Fill with 0"]
)

# Reset Filters Button
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset Filters", type="primary"):
    st.session_state.filters_applied = False
    st.rerun()


# ===========================
# MAIN
# ===========================
if uploaded_file:

    # Cache data loading
    @st.cache_data(ttl=3600)
    def load_data(file):
        return pd.read_csv(file)
    
    try:
        df = load_data(uploaded_file)
        st.session_state.original_df = df.copy()
        original_rows = len(df)
    except Exception as e:
        logger.log_error("Failed to load data", e)
        st.error(f"Error loading file: {str(e)}")
        st.stop()

    # Auto detect date - MUST be done BEFORE any data conversion
    date_col = None
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower() or "day" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
                date_col = col
                break
            except:
                pass

    # Store original date column for later use before any conversions
    original_date_col = date_col

    # ===========================
    # COLUMN ROLE DETECTION
    # ===========================
    try:
        roles = detect_column_roles(df)
        st.session_state.column_roles = roles
        display_column_roles(roles)
    except Exception as e:
        logger.log_error("Column role detection failed", e)
        roles = {}

    st.markdown("<br>", unsafe_allow_html=True)

    # ===========================
    # DATA QUALITY SCORE
    # ===========================
    st.markdown("## 📊 Data Quality Score")

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()

    completeness = 100 - ((missing_cells / total_cells) * 100) if total_cells > 0 else 0
    duplicate_score = 100 - ((duplicate_rows / len(df)) * 100) if len(df) > 0 else 100
    overall_score = round((completeness + duplicate_score) / 2, 2)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Completeness %", round(completeness, 2))
    c2.metric("Duplicate Free %", round(duplicate_score, 2))
    c3.metric("Overall Quality Score", overall_score)
    c4.metric("Total Columns", len(df.columns))

    st.markdown("---")

    # ===========================
    # EXECUTIVE SUMMARY
    # ===========================
    st.markdown("## 📝 Executive Summary")
    
    try:
        summary = generate_executive_summary(df, roles)
        st.info(summary)
    except Exception as e:
        logger.log_error("Executive summary failed", e)
        st.warning("Unable to generate executive summary")

    st.markdown("---")

    # ===========================
    # AI CLEANING SUGGESTIONS
    # ===========================
    st.markdown("## 🤖 AI Cleaning Suggestions")

    if model is not None:
        try:
            # Keep prompt small (avoid token overflow)
            summary = f"""
            Shape: {df.shape}
            Missing Values: {df.isnull().sum().sum()}
            Duplicate Rows: {duplicate_rows}
            Column Types: {df.dtypes.to_dict()}
            """

            prompt = f"""
            You are a professional data cleaning expert.
            Based on the following dataset summary,
            provide short and practical cleaning suggestions.

            {summary}
            """

            response = model.generate_content(prompt)

            if response.text:
                st.info(response.text)
            else:
                st.warning("No suggestions returned.")

        except Exception as e:
            logger.log_error("Gemini Runtime Error", e)
            st.error(f"Gemini Runtime Error: {e}")

    else:
        st.warning("Gemini API not configured.")

    # ===========================
    # CONTROLLED CLEANING
    # ===========================
    cleaned_df = df.copy()

    if trim_text:
        obj_cols = cleaned_df.select_dtypes(include="object").columns
        for col in obj_cols:
            cleaned_df[col] = cleaned_df[col].astype(str).str.strip().str.lower()

    # Restore date column if it was converted to numeric
    if original_date_col and original_date_col in df.columns:
        cleaned_df[original_date_col] = df[original_date_col].copy()
        date_col = original_date_col
    
    if convert_numeric:
        # Only convert non-date columns to numeric
        for col in cleaned_df.columns:
            if col != date_col:  # Skip the date column
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="ignore")

    if remove_duplicates:
        cleaned_df = cleaned_df.drop_duplicates()

    numeric_cols = cleaned_df.select_dtypes(include="number").columns

    if handle_missing == "Drop Rows":
        cleaned_df = cleaned_df.dropna()
    elif handle_missing == "Fill with Mean":
        for col in numeric_cols:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
    elif handle_missing == "Fill with Median":
        for col in numeric_cols:
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    elif handle_missing == "Fill with 0":
        cleaned_df = cleaned_df.fillna(0)

    if remove_outliers:
        cleaned_df = remove_outliers_iqr(cleaned_df)

    removed_rows = original_rows - len(cleaned_df)

    st.markdown("## 🛠 Preview After Cleaning")

    show_full = st.checkbox("Show Full Dataset")
    if show_full:
        st.dataframe(cleaned_df, use_container_width=True)
    else:
        st.dataframe(cleaned_df.head(10), use_container_width=True)

    st.success(f"Cleaning Summary: {removed_rows} rows removed.")

    # Download cleaned data
    csv = cleaned_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Cleaned Dataset", csv, "cleaned_data.csv")

    st.markdown("---")

    # ===========================
    # FILTERS
    # ===========================
    filtered_df = cleaned_df.copy()

    categorical_cols = filtered_df.select_dtypes(exclude="number").columns

    # Only allow filtering on low-cardinality columns
    for col in categorical_cols:
        unique_count = filtered_df[col].nunique()

        if unique_count <= 20:  # Only show filter if <= 20 unique values
            selected_values = st.sidebar.multiselect(
                f"Filter by {col}",
                options=filtered_df[col].unique(),
                default=list(filtered_df[col].unique())
            )

            # Only apply filter if something selected
            if selected_values:
                filtered_df = filtered_df[
                    filtered_df[col].isin(selected_values)
                ]

    # Date filter
    if date_col and not filtered_df.empty:
        min_date = filtered_df[date_col].min()
        max_date = filtered_df[date_col].max()

        if pd.notnull(min_date) and pd.notnull(max_date):
            date_range = st.sidebar.date_input(
                "Filter by Date Range",
                [min_date.date(), max_date.date()]
            )

            if len(date_range) == 2:
                filtered_df = filtered_df[
                    (filtered_df[date_col] >= pd.to_datetime(date_range[0])) &
                    (filtered_df[date_col] <= pd.to_datetime(date_range[1]))
                ]

    st.session_state.filters_applied = True

    # ===========================
    # DATASET PROFILE REPORT
    # ===========================
    try:
        generate_dataset_profile(filtered_df, roles)
    except Exception as e:
        logger.log_error("Dataset profile generation failed", e)
        st.error("Error generating profile report")

    st.markdown("---")

    # ===========================
    # UNIVERSAL KPI SECTION
    # ===========================
    st.markdown("## 🎯 KPIs")

    # Prepare JSON for caching
    df_json = filtered_df.to_json(orient='split')

    # Generate Auto KPIs first
    if auto_kpis:
        try:
            auto_kpis_list = generate_auto_kpis(df_json, date_col)
            
            st.markdown("### 🚀 Auto-Generated KPIs")
            
            cols = st.columns(4)
            for i, (name, value, kpi_type) in enumerate(auto_kpis_list[:12]):
                with cols[i % 4]:
                    st.metric(name, value)
        except Exception as e:
            logger.log_error("Auto KPI generation failed", e)
            st.error("Error generating auto KPIs")

    # Custom KPIs from user input
    if kpi_input:
        st.markdown("### 📝 Custom KPIs")
        kpis = [k.strip() for k in kpi_input.split(",")]
        cols = st.columns(3)

        for i, kpi in enumerate(kpis):
            try:
                value, kpi_type = generate_universal_kpi(kpi, df_json, date_col)
                if isinstance(value, dict):
                    for k, v in value.items():
                        cols[i % 3].metric(f"{kpi} - {k}", v)
                else:
                    cols[i % 3].metric(kpi, value)
            except Exception as e:
                logger.log_error(f"KPI generation failed for: {kpi}", e)
                cols[i % 3].metric(kpi, "Error")

    st.markdown("---")

    # ===========================
    # ENHANCED AUTO DASHBOARD
    # ===========================
    st.markdown("## 📈 Auto Dashboard")
    generate_enhanced_dashboard(filtered_df, date_col)

    st.markdown("---")

    # ===========================
    # CHATBOT
    # ===========================
    st.markdown("## 🤖 Chat with Your Data")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Ask about your dataset...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        if model:
            try:
                sample = filtered_df.head(50).to_string()
                stats = filtered_df.describe().to_string()

                prompt = f"""
                Dataset Columns: {list(filtered_df.columns)}
                Dataset Shape: {filtered_df.shape}
                Numeric Summary:
                {stats}
                Sample Data:
                {sample}

                Question: {user_input}

                Provide detailed numerical insights and recommendations.
                """

                response = model.generate_content(prompt)
                reply = response.text
            except Exception as e:
                logger.log_error("Chatbot error", e)
                reply = str(e)
        else:
            reply = "Gemini API not configured."

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

else:
    st.info("⬅️ Upload a CSV file to begin.")
    
    st.markdown("### 📌 Sample KPIs You Can Try:")
    st.code("""
total sales, average profit, total revenue
max quantity, min price, average rating
top product, unique customers
growth rate, total orders
category wise sales, sales trend, sales distribution
    """, language="text")
