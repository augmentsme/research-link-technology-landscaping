"""
Streamlit interface for topic classification analysis with interactive visualizations
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from inspect_ai.analysis.beta import messages_df, samples_df


class TopicClassificationAnalyzer:
    """Analyzer for topic classification evaluation results"""
    
    def __init__(self, eval_file: str, subfields_path: str = "/home/lcheng/oz318/research-link-technology-landscaping/data/subfields.json"):
        self.eval_file = eval_file
        self.subfields_path = subfields_path
        self.samples: Optional[pd.DataFrame] = None
        self.messages: Optional[pd.DataFrame] = None
        self.analysis_df: Optional[pd.DataFrame] = None
        self.id_to_name: Dict[str, str] = {}
        self.id_to_domain: Dict[str, str] = {}
        self.id_to_field: Dict[str, str] = {}
        self.subfields_data: list = []
        
    def load_data(self) -> None:
        """Load evaluation data and subfield mappings"""
        self.samples = samples_df(self.eval_file)
        self.messages = messages_df(self.eval_file)
        self._load_subfield_mappings()
        
    def _load_subfield_mappings(self) -> None:
        """Load subfield ID to name mappings and domain/field information"""
        try:
            with open(self.subfields_path, 'r') as f:
                self.subfields_data = json.load(f)
            
            self.id_to_name = {}
            self.id_to_domain = {}
            self.id_to_field = {}
            
            for subfield in self.subfields_data:
                canonical_id = subfield['id'].split('/')[-1]
                self.id_to_name[canonical_id] = subfield['display_name']
                
                # Extract domain information
                if 'domain' in subfield and subfield['domain']:
                    self.id_to_domain[canonical_id] = subfield['domain']['display_name']
                
                # Extract field information
                if 'field' in subfield and subfield['field']:
                    self.id_to_field[canonical_id] = subfield['field']['display_name']
                
        except Exception as e:
            st.error(f"Error loading subfield mappings: {e}")
            self.id_to_name = {}
            self.id_to_domain = {}
            self.id_to_field = {}
            
    def _extract_classification(self, response_text: str) -> Optional[str]:
        """Extract classification ID from model response"""
        if pd.isna(response_text):
            return None
        match = re.search(r'ANSWER:\s*([A-Za-z]?\d+)', response_text)
        return match.group(1) if match else None
        
    def process_responses(self) -> None:
        """Process model responses and extract predictions"""
        if self.samples is None or self.messages is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Extract predictions from model responses
        model_responses = self.messages[self.messages.source == "generate"].copy()
        model_responses['predicted_id'] = model_responses['content'].apply(self._extract_classification)
        
        # Merge with sample metadata
        self.analysis_df = self.samples.merge(
            model_responses[['sample_id', 'predicted_id']], 
            on='sample_id', 
            how='left'
        )
        
        # Add predicted names, domains, and fields
        if self.id_to_name:
            self.analysis_df['predicted_name'] = self.analysis_df['predicted_id'].map(self.id_to_name)
        if self.id_to_domain:
            self.analysis_df['predicted_domain'] = self.analysis_df['predicted_id'].map(self.id_to_domain)
        if self.id_to_field:
            self.analysis_df['predicted_field'] = self.analysis_df['predicted_id'].map(self.id_to_field)
        
    def get_prediction_distribution(self) -> pd.Series:
        """Get distribution of predictions by subfield"""
        if self.analysis_df is None:
            raise ValueError("Data not processed. Call process_responses() first.")
            
        if 'predicted_name' in self.analysis_df.columns:
            return self.analysis_df['predicted_name'].value_counts()
        else:
            return self.analysis_df['predicted_id'].value_counts()


def main():
    """Simple Streamlit application showing only label distribution"""
    st.set_page_config(
        page_title="Label Distribution Analysis",
        page_icon="📊",
        layout="centered"
    )
    
    st.title("� Label Distribution Analysis")
    st.markdown("---")
    
    # File selection
    eval_files = list(Path("logs").glob("*.eval"))
    if not eval_files:
        st.error("No evaluation files found in the logs directory!")
        return
        
    selected_file = st.selectbox(
        "Select Evaluation File",
        options=[f.name for f in eval_files],
        index=0
    )
    
    eval_path = f"logs/{selected_file}"
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state or st.session_state.get('current_file') != selected_file:
        with st.spinner("Loading and processing data..."):
            try:
                analyzer = TopicClassificationAnalyzer(eval_path)
                analyzer.load_data()
                analyzer.process_responses()
                st.session_state.analyzer = analyzer
                st.session_state.current_file = selected_file
                st.success("Data loaded successfully!")
            except Exception as e:
                st.error(f"Error loading data: {e}")
                return
    else:
        analyzer = st.session_state.analyzer
    
    # Get prediction distribution
    prediction_dist = analyzer.get_prediction_distribution()
    
    # Check if we have any predictions
    if len(prediction_dist) == 0:
        st.error("No valid predictions found in the dataset. Please check your evaluation file.")
        return
    
    # Show basic stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", len(analyzer.analysis_df))
    with col2:
        st.metric("Unique Labels", len(prediction_dist))
    with col3:
        st.metric("Most Common Label", prediction_dist.index[0])
    
    st.markdown("---")
    
    # Create pie chart for label distribution
    st.subheader("Label Distribution")
    
    # Show top 10 labels + others
    top_10 = prediction_dist.head(10)
    others_count = prediction_dist.iloc[10:].sum()
    
    if others_count > 0:
        pie_data = pd.concat([top_10, pd.Series([others_count], index=['Others'])])
    else:
        pie_data = top_10
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set3(range(len(pie_data)))
    wedges, texts, autotexts = ax.pie(
        pie_data.values, 
        labels=pie_data.index, 
        autopct='%1.1f%%', 
        colors=colors, 
        startangle=90
    )
    
    # Improve text readability
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title('Distribution of Labels in Dataset', fontsize=16, fontweight='bold')
    st.pyplot(fig)
    
    # Show detailed table
    st.subheader("Detailed Label Counts")
    label_df = pd.DataFrame({
        'Label': prediction_dist.index,
        'Count': prediction_dist.values,
        'Percentage': (prediction_dist.values / prediction_dist.sum() * 100).round(2)
    })
    st.dataframe(label_df, use_container_width=True, height=400)


if __name__ == "__main__":
    main()
