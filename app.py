import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from docx import Document
from docx.shared import Inches
import io
import base64
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Advanced Data Analysis Tool",
    page_icon="📊",
    layout="wide"
)

def perform_analysis_and_generate_report(df):
    """
    Perform comprehensive data analysis and generate a Word report
    """
    # Initialize report buffer
    buffer = io.BytesIO()
    
    # Detect column types
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Statistical Analysis
    stats_summary = {}
    correlation_insights = []
    
    # Analyze numerical columns
    numerical_stats = {}
    for col in numerical_cols:
        if df[col].notna().sum() > 0:  # Only analyze columns with data
            stats = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'top_5_values': df[col].nlargest(5).tolist()
            }
            numerical_stats[col] = stats
    
    # Analyze categorical columns
    categorical_stats = {}
    for col in categorical_cols:
        if df[col].notna().sum() > 0:  # Only analyze columns with data
            value_counts = df[col].value_counts().head(5)
            categorical_stats[col] = {
                'top_5_categories': value_counts.to_dict(),
                'unique_count': df[col].nunique(),
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None
            }
    
    # Correlation Analysis (only for numerical columns with more than 1 column)
    strong_correlations = []
    if len(numerical_cols) >= 2:
        corr_matrix = df[numerical_cols].corr()
        for i in range(len(numerical_cols)):
            for j in range(i+1, len(numerical_cols)):
                correlation = corr_matrix.iloc[i, j]
                if abs(correlation) > 0.7 and not np.isnan(correlation):
                    strong_correlations.append({
                        'col1': numerical_cols[i],
                        'col2': numerical_cols[j],
                        'correlation': correlation
                    })
    
    # Generate Dynamic Narrative
    executive_summary = generate_executive_summary(df, numerical_stats, categorical_stats, strong_correlations)
    
    # Generate Recommendations
    recommendations = generate_recommendations(df, numerical_stats, categorical_stats, strong_correlations)
    
    # Create Visualizations
    fig1, fig2 = create_visualizations(df, numerical_cols, categorical_cols)
    
    # Generate Word Document
    doc = Document()
    
    # Add Title
    title = doc.add_heading('Automated Data Analysis Report', 0)
    title.alignment = 1  # Center alignment
    
    # Add Executive Summary
    doc.add_heading('Executive Summary', level=1)
    doc.add_paragraph(executive_summary)
    
    # Add Key Findings
    doc.add_heading('Key Findings', level=1)
    
    # Numerical Statistics Table
    if numerical_stats:
        doc.add_heading('Numerical Analysis', level=2)
        table = doc.add_table(rows=1, cols=6)
        table.style = 'Light Grid Accent 1'
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = 'Column'
        hdr_cells[1].text = 'Mean'
        hdr_cells[2].text = 'Median'
        hdr_cells[3].text = 'Std Dev'
        hdr_cells[4].text = 'Min'
        hdr_cells[5].text = 'Max'
        
        for col, stats in numerical_stats.items():
            row_cells = table.add_row().cells
            row_cells[0].text = col
            row_cells[1].text = f"{stats['mean']:.2f}"
            row_cells[2].text = f"{stats['median']:.2f}"
            row_cells[3].text = f"{stats['std']:.2f}"
            row_cells[4].text = f"{stats['min']:.2f}"
            row_cells[5].text = f"{stats['max']:.2f}"
    
    # Categorical Statistics
    if categorical_stats:
        doc.add_heading('Categorical Analysis', level=2)
        for col, stats in categorical_stats.items():
            doc.add_paragraph(f"Column '{col}': {stats['unique_count']} unique values")
            doc.add_paragraph(f"Most frequent value: {stats['most_frequent']}")
    
    # Add Recommendations
    doc.add_heading('Recommendations', level=1)
    for rec in recommendations:
        doc.add_paragraph(rec, style='List Bullet')
    
    # Add Visualizations
    doc.add_heading('Data Visualizations', level=1)
    
    # Save plots and add to document
    if fig1 is not None:
        img_buffer1 = io.BytesIO()
        fig1.savefig(img_buffer1, format='png', dpi=300, bbox_inches='tight')
        img_buffer1.seek(0)
        doc.add_paragraph('Distribution Analysis:')
        doc.add_picture(img_buffer1, width=Inches(6))
        plt.close(fig1)
    
    if fig2 is not None:
        img_buffer2 = io.BytesIO()
        fig2.savefig(img_buffer2, format='png', dpi=300, bbox_inches='tight')
        img_buffer2.seek(0)
        doc.add_paragraph('Category Frequency Analysis:')
        doc.add_picture(img_buffer2, width=Inches(6))
        plt.close(fig2)
    
    # Save document to buffer
    doc.save(buffer)
    buffer.seek(0)
    
    return buffer

def generate_executive_summary(df, numerical_stats, categorical_stats, strong_correlations):
    """Generate dynamic narrative based on analysis results"""
    summary = f"This report analyzes a dataset containing {len(df)} records and {len(df.columns)} columns. "
    
    if numerical_stats:
        # Find the primary numerical column (first one with highest variance or most data)
        primary_num_col = None
        max_variance = 0
        for col, stats in numerical_stats.items():
            if stats['std'] > max_variance:
                max_variance = stats['std']
                primary_num_col = col
        
        if primary_num_col:
            stats = numerical_stats[primary_num_col]
            mean_val = stats['mean']
            median_val = stats['median']
            
            summary += f"The primary numerical variable '{primary_num_col}' shows "
            
            # Check for skewness
            if abs(mean_val - median_val) / stats['std'] > 0.5:
                if mean_val > median_val:
                    summary += "a right-skewed distribution, indicating some high-value outliers. "
                else:
                    summary += "a left-skewed distribution, indicating some low-value outliers. "
            else:
                summary += "a relatively normal distribution. "
    
    # Add correlation insights
    if strong_correlations:
        summary += f"Strong correlations were identified between {len(strong_correlations)} pairs of variables. "
        strongest_corr = max(strong_correlations, key=lambda x: abs(x['correlation']))
        summary += f"The strongest relationship exists between '{strongest_corr['col1']}' and '{strongest_corr['col2']}' (r={strongest_corr['correlation']:.2f}). "
    
    # Add categorical insights
    if categorical_stats:
        primary_cat_col = list(categorical_stats.keys())[0]
        most_frequent = categorical_stats[primary_cat_col]['most_frequent']
        unique_count = categorical_stats[primary_cat_col]['unique_count']
        summary += f"The categorical variable '{primary_cat_col}' contains {unique_count} unique categories, with '{most_frequent}' being the most frequent."
    
    return summary

def generate_recommendations(df, numerical_stats, categorical_stats, strong_correlations):
    """Generate actionable recommendations based on analysis"""
    recommendations = []
    
    # Recommendations based on numerical analysis
    for col, stats in numerical_stats.items():
        mean_val = stats['mean']
        
        # Check if this looks like a satisfaction/rating column
        if any(keyword in col.lower() for keyword in ['satisfaction', 'rating', 'score', 'quality']):
            if mean_val < 3.5:  # Assuming 1-5 scale
                recommendations.append(f"Focus on improving {col.lower()} - current average of {mean_val:.2f} indicates significant room for improvement.")
            elif mean_val > 4.0:
                recommendations.append(f"Maintain high standards in {col.lower()} - current average of {mean_val:.2f} shows strong performance.")
        
        # Check for high variability
        cv = stats['std'] / stats['mean'] if stats['mean'] != 0 else 0
        if cv > 0.5:
            recommendations.append(f"Investigate high variability in '{col}' - consider standardizing processes to reduce inconsistency.")
    
    # Recommendations based on correlations
    for corr in strong_correlations:
        if corr['correlation'] > 0.7:
            recommendations.append(f"Leverage the strong positive relationship between '{corr['col1']}' and '{corr['col2']}' - increasing {corr['col1']} may positively impact {corr['col2']}.")
        elif corr['correlation'] < -0.7:
            recommendations.append(f"Address the strong negative relationship between '{corr['col1']}' and '{corr['col2']}' - monitor these metrics carefully as they move in opposite directions.")
    
    # Recommendations based on categorical analysis
    for col, stats in categorical_stats.items():
        if stats['unique_count'] > len(df) * 0.8:  # High cardinality
            recommendations.append(f"Consider grouping categories in '{col}' - {stats['unique_count']} unique values may be too granular for analysis.")
    
    # General recommendations
    if len(df) < 100:
        recommendations.append("Consider collecting more data - current sample size may limit the reliability of statistical conclusions.")
    
    if not recommendations:
        recommendations.append("Continue monitoring key metrics and consider collecting additional relevant data points for deeper insights.")
    
    return recommendations

def create_visualizations(df, numerical_cols, categorical_cols):
    """Create dynamic visualizations based on data"""
    fig1, fig2 = None, None
    
    # Create histogram for primary numerical column
    if numerical_cols:
        primary_num_col = numerical_cols[0]  # Use first numerical column
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        # Remove any infinite or extremely large values
        clean_data = df[primary_num_col].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(clean_data) > 0:
            ax1.hist(clean_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title(f'Distribution of {primary_num_col}', fontsize=14, fontweight='bold')
            ax1.set_xlabel(primary_num_col, fontsize=12)
            ax1.set_ylabel('Frequency', fontsize=12)
            ax1.grid(axis='y', alpha=0.3)
            
            # Add statistical annotations
            mean_val = clean_data.mean()
            median_val = clean_data.median()
            ax1.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            ax1.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
            ax1.legend()
    
    # Create bar chart for primary categorical column
    if categorical_cols:
        primary_cat_col = categorical_cols[0]  # Use first categorical column
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        value_counts = df[primary_cat_col].value_counts().head(10)  # Top 10 categories
        
        if len(value_counts) > 0:
            bars = ax2.bar(range(len(value_counts)), value_counts.values, 
                          color='lightcoral', alpha=0.8, edgecolor='black')
            ax2.set_title(f'Top Categories in {primary_cat_col}', fontsize=14, fontweight='bold')
            ax2.set_xlabel(primary_cat_col, fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_xticks(range(len(value_counts)))
            ax2.set_xticklabels(value_counts.index, rotation=45, ha='right')
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, value_counts.values):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + value*0.01,
                        str(value), ha='center', va='bottom')
        
        plt.tight_layout()
    
    return fig1, fig2

def main():
    """Main Streamlit application"""
    # Title and description
    st.title("📊 Advanced Data Analysis Tool")
    st.markdown("Upload a CSV file to generate comprehensive analysis and downloadable report")
    
    # Sidebar with instructions
    with st.sidebar:
        st.header("Instructions")
        st.markdown("""
        1. Upload your CSV file
        2. Click 'Generate Report'
        3. View the analysis results
        4. Download the Word document report
        
        **Features:**
        - Automatic statistical analysis
        - Dynamic narrative generation
        - Actionable recommendations
        - Professional visualizations
        - Downloadable Word report
        """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Display basic info about the dataset
            st.subheader("Dataset Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Show first few rows
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
            
            # Generate Report Button
            if st.button("Generate Report", type="primary"):
                with st.spinner("Analyzing data and generating report..."):
                    try:
                        # Perform analysis and generate report
                        report_buffer = perform_analysis_and_generate_report(df)
                        
                        st.success("Report generated successfully!")
                        
                        # Provide download button
                        st.download_button(
                            label="📄 Download Analysis Report (Word Document)",
                            data=report_buffer.getvalue(),
                            file_name="data_analysis_report.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                        
                        # Display some key insights in the app
                        st.subheader("Key Insights Preview")
                        
                        # Quick stats
                        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.info(f"**Numerical Columns:** {len(numerical_cols)}")
                            if numerical_cols:
                                st.write("• " + "\n• ".join(numerical_cols[:5]))
                        
                        with col2:
                            st.info(f"**Categorical Columns:** {len(categorical_cols)}")
                            if categorical_cols:
                                st.write("• " + "\n• ".join(categorical_cols[:5]))
                        
                    except Exception as e:
                        st.error(f"Error generating report: {str(e)}")
                        st.info("Please ensure your CSV file is properly formatted and contains analyzable data.")
        
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
            st.info("Please check that your file is a valid CSV format.")
    
    else:
        st.info("Please upload a CSV file to begin analysis.")
        
        # Show example of what the tool can do
        with st.expander("See example analysis"):
            st.markdown("""
            **This tool will automatically:**
            
            🔍 **Analyze your data:**
            - Detect numerical and categorical columns
            - Calculate statistical measures (mean, median, std dev)
            - Identify correlations and outliers
            - Generate frequency distributions
            
            📝 **Create narratives:**
            - Dynamic executive summary
            - Data-driven insights
            - Statistical interpretations
            
            💡 **Provide recommendations:**
            - Actionable business insights
            - Process improvements
            - Data quality suggestions
            
            📊 **Generate visualizations:**
            - Distribution histograms
            - Category frequency charts
            - Professional styling
            
            📄 **Produce Word report:**
            - Executive summary
            - Statistical tables
            - Visualizations
            - Recommendations
            """)

if __name__ == "__main__":
    main()
