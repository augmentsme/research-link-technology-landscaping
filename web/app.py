"""
Streamlit interface for topic classification analysis with interactive visualizations
"""

import datetime
import json
import re
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from inspect_ai.analysis.beta import messages_df, samples_df
import numpy as np


class TopicClassificationAnalyzer:
    """Analyzer for topic classification evaluation results"""
    def __init__(self, eval_file: str, base_data_dir: str = "/home/lcheng/oz318/research-link-technology-landscaping/data"):
        self.eval_file = eval_file
        self.base_data_dir = base_data_dir
        self.samples: Optional[pd.DataFrame] = None
        self.messages: Optional[pd.DataFrame] = None
        self.analysis_df: Optional[pd.DataFrame] = None
        self.id_to_name: Dict[str, str] = {}
        self.id_to_domain: Dict[str, str] = {}
        self.id_to_field: Dict[str, str] = {}
        self.classification_data: list = []
        self.classification_type: Optional[str] = None
        self.grants_data: Optional[pd.DataFrame] = None
        
    def load_data(self) -> None:
        """Load evaluation data and classification mappings"""
        self.samples = samples_df(self.eval_file)
        self.messages = messages_df(self.eval_file)
        
        # Detect classification type from metadata
        if 'metadata_classification_type' in self.samples.columns:
            self.classification_type = self.samples['metadata_classification_type'].iloc[0]
        else:
            # Fallback to subfields if not specified
            self.classification_type = 'subfields'
            
        self._load_classification_mappings()
        self._load_grants_data()
        
    def _load_classification_mappings(self) -> None:
        """Load classification ID to name mappings and hierarchical information"""
        try:
            # Determine the appropriate data file based on classification type
            classification_file = f"{self.classification_type}.json"
            classification_path = Path(self.base_data_dir) / classification_file
            
            with open(classification_path, 'r') as f:
                self.classification_data = json.load(f)
            
            self.id_to_name = {}
            self.id_to_domain = {}
            self.id_to_field = {}
            
            for item in self.classification_data:
                canonical_id = item['id'].split('/')[-1]
                self.id_to_name[canonical_id] = item['display_name']
                
                # Extract domain information (available for fields, subfields, topics)
                if 'domain' in item and item['domain']:
                    self.id_to_domain[canonical_id] = item['domain']['display_name']
                
                # Extract field information (available for subfields, topics)
                if 'field' in item and item['field']:
                    self.id_to_field[canonical_id] = item['field']['display_name']
                
        except Exception as e:
            st.error(f"Error loading {self.classification_type} mappings: {e}")
            self.id_to_name = {}
            self.id_to_domain = {}
            self.id_to_field = {}
            
    def _load_grants_data(self) -> None:
        """Load original grants data for funding information"""
        try:
            grants_path = Path(self.base_data_dir) / "active_grants.json"
            with open(grants_path, 'r') as f:
                grants_raw = json.load(f)
            
            # Convert to DataFrame for easier manipulation
            self.grants_data = pd.DataFrame(grants_raw)
            
        except Exception as e:
            st.error(f"Error loading grants data: {e}")
            self.grants_data = None
            
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
        
        # Add funding information if grants data is available
        if self.grants_data is not None:
            # Extract grant ID from metadata and merge with grants data
            grant_funding = self.grants_data[['id', 'funding_amount']].copy()
            grant_funding = grant_funding.rename(columns={'id': 'metadata_grant_id'})
            
            self.analysis_df = self.analysis_df.merge(
                grant_funding,
                on='metadata_grant_id',
                how='left'
            )
        
    def get_prediction_distribution(self) -> pd.Series:
        """Get distribution of predictions by classification type"""
        if self.analysis_df is None:
            raise ValueError("Data not processed. Call process_responses() first.")
            
        if 'predicted_name' in self.analysis_df.columns:
            return self.analysis_df['predicted_name'].value_counts()
        else:
            return self.analysis_df['predicted_id'].value_counts()
    
    def get_model_info(self) -> Dict[str, any]:
        """Extract model and token usage information"""
        if self.samples is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        models_used = set()
        total_tokens = {}
        total_samples = len(self.samples)
        
        # Parse model usage from samples
        for usage_str in self.samples['model_usage'].dropna():
            if usage_str and usage_str != '':
                try:
                    usage = json.loads(usage_str)
                    for model, tokens in usage.items():
                        models_used.add(model)
                        if model not in total_tokens:
                            total_tokens[model] = {
                                'input_tokens': 0, 
                                'output_tokens': 0, 
                                'total_tokens': 0,
                                'reasoning_tokens': 0
                            }
                        total_tokens[model]['input_tokens'] += tokens.get('input_tokens', 0)
                        total_tokens[model]['output_tokens'] += tokens.get('output_tokens', 0)
                        total_tokens[model]['total_tokens'] += tokens.get('total_tokens', 0)
                        total_tokens[model]['reasoning_tokens'] += tokens.get('reasoning_tokens', 0)
                except json.JSONDecodeError:
                    continue
        
        # Calculate total time
        total_time = self.samples['total_time'].sum()
        avg_time_per_sample = self.samples['total_time'].mean()
        
        # Extract evaluation metadata
        eval_id = self.samples['eval_id'].iloc[0] if len(self.samples) > 0 else "Unknown"
        
        return {
            'models_used': list(models_used),
            'total_tokens': total_tokens,
            'total_samples': total_samples,
            'total_time': total_time,
            'avg_time_per_sample': avg_time_per_sample,
            'eval_id': eval_id
        }
    
    def get_file_info(self) -> Dict[str, any]:
        """Get information about the evaluation file"""
        file_path = Path(self.eval_file)
        file_size = file_path.stat().st_size
        file_modified = file_path.stat().st_mtime
        
        return {
            'file_name': file_path.name,
            'file_size_mb': file_size / (1024 * 1024),
            'file_modified': file_modified
        }
    
    def get_hierarchical_distribution(self) -> Dict[str, pd.Series]:
        """Get distribution by domain and field (when available)"""
        if self.analysis_df is None:
            raise ValueError("Data not processed. Call process_responses() first.")
        
        distributions = {}
        
        # Domain distribution (available for fields, subfields, topics)
        if 'predicted_domain' in self.analysis_df.columns:
            domain_dist = self.analysis_df['predicted_domain'].value_counts()
            distributions['domain'] = domain_dist
        
        # Field distribution (available for subfields, topics)
        if 'predicted_field' in self.analysis_df.columns:
            field_dist = self.analysis_df['predicted_field'].value_counts()
            distributions['field'] = field_dist
            
        return distributions
    
    def get_funding_distribution(self) -> pd.Series:
        """Get distribution of total funding by classification type"""
        if self.analysis_df is None:
            raise ValueError("Data not processed. Call process_responses() first.")
        
        if 'funding_amount' not in self.analysis_df.columns:
            return pd.Series(dtype=float)
        
        # Group by predicted name and sum funding amounts
        if 'predicted_name' in self.analysis_df.columns:
            funding_dist = self.analysis_df.groupby('predicted_name')['funding_amount'].sum().sort_values(ascending=False)
        else:
            funding_dist = self.analysis_df.groupby('predicted_id')['funding_amount'].sum().sort_values(ascending=False)
        
        return funding_dist
    
    def get_hierarchical_funding_distribution(self) -> Dict[str, pd.Series]:
        """Get funding distribution by domain and field (when available)"""
        if self.analysis_df is None:
            raise ValueError("Data not processed. Call process_responses() first.")
        
        if 'funding_amount' not in self.analysis_df.columns:
            return {}
        
        distributions = {}
        
        # Domain funding distribution (available for fields, subfields, topics)
        if 'predicted_domain' in self.analysis_df.columns:
            domain_funding = self.analysis_df.groupby('predicted_domain')['funding_amount'].sum().sort_values(ascending=False)
            distributions['domain'] = domain_funding
        
        # Field funding distribution (available for subfields, topics)
        if 'predicted_field' in self.analysis_df.columns:
            field_funding = self.analysis_df.groupby('predicted_field')['funding_amount'].sum().sort_values(ascending=False)
            distributions['field'] = field_funding
            
        return distributions


def main():
    """Streamlit application for analyzing classification evaluation results"""
    st.set_page_config(
        page_title="Classification Analysis Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Classification Analysis Dashboard")
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
    
    # Get evaluation metadata
    model_info = analyzer.get_model_info()
    file_info = analyzer.get_file_info()
    prediction_dist = analyzer.get_prediction_distribution()
    hierarchical_dist = analyzer.get_hierarchical_distribution()
    funding_dist = analyzer.get_funding_distribution()
    hierarchical_funding_dist = analyzer.get_hierarchical_funding_distribution()
    
    # Check if we have any predictions
    if len(prediction_dist) == 0:
        st.error("No valid predictions found in the dataset. Please check your evaluation file.")
        st.write(f"Debug info: Classification type = {analyzer.classification_type}")
        st.write(f"Debug info: Analysis df shape = {analyzer.analysis_df.shape if analyzer.analysis_df is not None else 'None'}")
        st.write(f"Debug info: ID mappings = {len(analyzer.id_to_name)}")
        if analyzer.analysis_df is not None and len(analyzer.analysis_df) > 0:
            st.write(f"Debug info: Sample predicted IDs = {analyzer.analysis_df['predicted_id'].head().tolist()}")
            if 'predicted_name' in analyzer.analysis_df.columns:
                st.write(f"Debug info: Sample predicted names = {analyzer.analysis_df['predicted_name'].head().tolist()}")
        return
    
    # Display evaluation metadata
    st.subheader("📊 Evaluation Overview")
    
    # File information
    file_date = datetime.datetime.fromtimestamp(file_info['file_modified'])
    classification_type_display = analyzer.classification_type.title() if analyzer.classification_type else "Unknown"
    st.info(f"**File:** {file_info['file_name']} | **Classification Type:** {classification_type_display} | **Size:** {file_info['file_size_mb']:.1f} MB | **Modified:** {file_date.strftime('%Y-%m-%d %H:%M:%S')} | **Eval ID:** {model_info['eval_id'][:8]}...")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Dataset Information")
        st.metric("Total Samples", model_info['total_samples'])
        st.metric("Unique Labels Predicted", len(prediction_dist))
        st.metric("Most Common Label", prediction_dist.index[0])
        
        if model_info['total_time'] > 0:
            st.metric("Total Evaluation Time", f"{model_info['total_time']:.1f}s")
            st.metric("Avg Time per Sample", f"{model_info['avg_time_per_sample']:.2f}s")
    
    with col2:
        st.markdown("##### Model & Token Usage")
        
        # Display model information
        if model_info['models_used']:
            st.write("**Models Used:**")
            for model in model_info['models_used']:
                st.write(f"• {model}")
            
            # Token usage information
            for model, tokens in model_info['total_tokens'].items():
                st.markdown(f"**{model} Token Usage:**")
                col2a, col2b = st.columns(2)
                with col2a:
                    st.metric("Input Tokens", f"{tokens['input_tokens']:,}")
                    st.metric("Output Tokens", f"{tokens['output_tokens']:,}")
                with col2b:
                    st.metric("Total Tokens", f"{tokens['total_tokens']:,}")
                    if tokens['reasoning_tokens'] > 0:
                        st.metric("Reasoning Tokens", f"{tokens['reasoning_tokens']:,}")
                
                # Cost estimation (approximate for o4-mini)
                if 'o4-mini' in model:
                    # Rough cost estimate: ~$0.50 per 1M tokens
                    estimated_cost = (tokens['total_tokens'] / 1_000_000) * 0.50
                    st.metric("Estimated Cost", f"${estimated_cost:.3f}")
    
    st.markdown("---")
    
    # Summary statistics box
    with st.container():
        st.subheader("📈 Key Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        total_tokens_all = sum(tokens['total_tokens'] for tokens in model_info['total_tokens'].values())
        
        with col1:
            st.metric("Total Tokens Used", f"{total_tokens_all:,}")
        with col2:
            st.metric("Total Evaluation Time", f"{model_info['total_time']:.1f}s")
        with col3:
            st.metric("Throughput", f"{model_info['total_samples']/model_info['total_time']:.1f} samples/sec")
        with col4:
            # Extract number of available labels from metadata if available
            num_labels = analyzer.samples['metadata_num_labels'].iloc[0] if 'metadata_num_labels' in analyzer.samples.columns else "Unknown"
            st.metric("Label Coverage", f"{len(prediction_dist)}/{num_labels}")
    
    st.markdown("---")
    
    # Create pie chart for label distribution
    classification_display = analyzer.classification_type.title() if analyzer.classification_type else "Label"
    st.subheader(f"🎯 {classification_display} Distribution")
    
    # Show top 10 labels + others
    top_10 = prediction_dist.head(10)
    others_count = prediction_dist.iloc[10:].sum()
    
    if others_count > 0:
        pie_data = pd.concat([top_10, pd.Series([others_count], index=['Others'])])
    else:
        pie_data = top_10
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
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
        
        ax.set_title(f'Distribution of {classification_display}s in Dataset', fontsize=16, fontweight='bold')
        st.pyplot(fig)
    
    with col2:
        st.markdown(f"##### Top 5 {classification_display}s")
        top_5_df = pd.DataFrame({
            classification_display: prediction_dist.head(5).index,
            'Count': prediction_dist.head(5).values,
            'Percentage': (prediction_dist.head(5).values / prediction_dist.sum() * 100).round(1)
        })
        st.dataframe(top_5_df, use_container_width=True)
    
    # Show detailed table
    st.subheader(f"📋 Detailed {classification_display} Counts")
    label_df = pd.DataFrame({
        classification_display: prediction_dist.index,
        'Count': prediction_dist.values,
        'Percentage': (prediction_dist.values / prediction_dist.sum() * 100).round(2)
    })
    st.dataframe(label_df, use_container_width=True, height=400)
    
    # Show hierarchical distributions if available
    if hierarchical_dist:
        st.markdown("---")
        st.subheader("🏗️ Hierarchical Distribution")
        
        # Create columns for hierarchical charts
        hierarchy_cols = st.columns(len(hierarchical_dist))
        
        for i, (level, distribution) in enumerate(hierarchical_dist.items()):
            with hierarchy_cols[i]:
                st.markdown(f"##### By {level.title()}")
                
                # Create a bar chart for hierarchical distribution
                fig, ax = plt.subplots(figsize=(8, 6))
                top_items = distribution.head(10)  # Show top 10
                
                bars = ax.barh(range(len(top_items)), top_items.values)
                ax.set_yticks(range(len(top_items)))
                ax.set_yticklabels(top_items.index)
                ax.set_xlabel('Count')
                ax.set_title(f'Top {level.title()}s')
                
                # Add value labels on bars
                for j, (bar, value) in enumerate(zip(bars, top_items.values)):
                    ax.text(value + 0.1, j, str(value), va='center')
                
                plt.tight_layout()
                st.pyplot(fig)
    
    # Show funding-based distributions if funding data is available
    if len(funding_dist) > 0:
        st.markdown("---")
        st.subheader(f"💰 {classification_display} Distribution by Funding Amount")
        
        # Format funding amounts for display
        def format_currency(amount):
            if amount >= 1_000_000:
                return f"${amount/1_000_000:.1f}M"
            elif amount >= 1_000:
                return f"${amount/1_000:.0f}K"
            else:
                return f"${amount:.0f}"
        
        # Show top 10 funding recipients + others
        top_10_funding = funding_dist.head(10)
        others_funding = funding_dist.iloc[10:].sum()
        
        if others_funding > 0:
            funding_pie_data = pd.concat([top_10_funding, pd.Series([others_funding], index=['Others'])])
        else:
            funding_pie_data = top_10_funding
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 8))
            # Use log scale for color mapping to highlight differences
            log_values = np.log1p(funding_pie_data.values)
            normed = (log_values - log_values.min()) / (log_values.max() - log_values.min() + 1e-9)
            colors = plt.cm.viridis(normed)
            
            wedges, texts, autotexts = ax.pie(
                funding_pie_data.values, 
                labels=funding_pie_data.index, 
                autopct=lambda pct: format_currency(pct * funding_pie_data.sum() / 100),
                colors=colors, 
                startangle=90
            )
            
            # Improve text readability
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title(f'Funding Distribution by {classification_display}', fontsize=16, fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            st.markdown(f"##### Top 5 {classification_display}s by Funding")
            top_5_funding_df = pd.DataFrame({
                classification_display: funding_dist.head(5).index,
                'Total Funding': [format_currency(x) for x in funding_dist.head(5).values],
                'Percentage': (funding_dist.head(5).values / funding_dist.sum() * 100).round(1)
            })
            st.dataframe(top_5_funding_df, use_container_width=True)
        
        # Show detailed funding table
        st.subheader(f"📋 Detailed {classification_display} Funding")
        funding_df = pd.DataFrame({
            classification_display: funding_dist.index,
            'Total Funding': [format_currency(x) for x in funding_dist.values],
            'Raw Amount': funding_dist.values,
            'Percentage': (funding_dist.values / funding_dist.sum() * 100).round(2)
        })
        st.dataframe(funding_df, use_container_width=True, height=400)
        
        # Show hierarchical funding distributions if available
        if hierarchical_funding_dist:
            st.markdown("---")
            st.subheader("🏗️ Hierarchical Funding Distribution")
            
            # Create columns for hierarchical funding charts
            hierarchy_funding_cols = st.columns(len(hierarchical_funding_dist))
            
            for i, (level, distribution) in enumerate(hierarchical_funding_dist.items()):
                with hierarchy_funding_cols[i]:
                    st.markdown(f"##### By {level.title()} (Funding)")
                    
                    # Create a bar chart for hierarchical funding distribution
                    fig, ax = plt.subplots(figsize=(8, 6))
                    top_items = distribution.head(10)  # Show top 10
                    
                    bars = ax.barh(range(len(top_items)), top_items.values)
                    ax.set_yticks(range(len(top_items)))
                    ax.set_yticklabels(top_items.index)
                    ax.set_xlabel('Total Funding ($)')
                    ax.set_title(f'Top {level.title()}s by Funding')
                    
                    # Add value labels on bars
                    for j, (bar, value) in enumerate(zip(bars, top_items.values)):
                        ax.text(value + max(top_items.values) * 0.01, j, format_currency(value), va='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig)


if __name__ == "__main__":
    main()
