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
from anzsrc.data import load_anzsrc




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
        self.template_type: Optional[str] = None
        self.is_cot: bool = False
        self.is_multiple_correct: bool = False
        
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
        
        # Detect template type and CoT/multiple_correct settings from metadata
        if 'metadata_template_type' in self.samples.columns:
            self.template_type = self.samples['metadata_template_type'].iloc[0]
            self.is_cot = 'COT' in self.template_type
            self.is_multiple_correct = 'MULTIPLE' in self.template_type
        else:
            # Fallback defaults
            self.template_type = 'SINGLE_ANSWER'
            self.is_cot = False
            self.is_multiple_correct = False
            
        self._load_classification_mappings()
        self._load_grants_data()
        
    def _load_classification_mappings(self) -> None:
        """Load classification ID to name mappings and hierarchical information"""
        try:
            # Check if this is an ANZSRC classification
            anzsrc_types = ['for_divisions', 'for_groups', 'seo_divisions', 'seo_groups']
            
            if self.classification_type in anzsrc_types:
                self._load_anzsrc_mappings()
            else:
                # Handle OpenAlex classifications (domains, fields, subfields, topics)
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
    
    def _load_anzsrc_mappings(self) -> None:
        """Load ANZSRC classification mappings from the combined DataFrame."""
        try:
            # Load ANZSRC data
            anzsrc_df = load_anzsrc()
            
            self.id_to_name = {}
            self.id_to_domain = {}
            self.id_to_field = {}
            self.classification_data = []
            
            if self.classification_type == 'for_divisions':
                # Get FoR divisions
                for_divisions = anzsrc_df[
                    (anzsrc_df.index.get_level_values('classification') == 'FoR') & 
                    (anzsrc_df['level'] == 'division')
                ]
                
                for (classification, division_code, _), row in for_divisions.iterrows():
                    division_code_str = str(division_code)  # Ensure string format
                    self.id_to_name[division_code_str] = row['division_name']
                    self.id_to_domain[division_code_str] = 'FoR'
                    # Store as dict for compatibility
                    self.classification_data.append({
                        'id': division_code_str,
                        'display_name': row['division_name'],
                        'definition': row['definition'],
                        'classification': classification
                    })
            
            elif self.classification_type == 'for_groups':
                # Get FoR groups
                for_groups = anzsrc_df[
                    (anzsrc_df.index.get_level_values('classification') == 'FoR') & 
                    (anzsrc_df['level'] == 'group')
                ]
                
                for (classification, division_code, group_code), row in for_groups.iterrows():
                    group_code_str = str(group_code)  # Ensure string format
                    self.id_to_name[group_code_str] = row['group_name']
                    self.id_to_domain[group_code_str] = 'FoR'
                    self.id_to_field[group_code_str] = row['division_name']
                    # Store as dict for compatibility
                    self.classification_data.append({
                        'id': group_code_str,
                        'display_name': row['group_name'],
                        'definition': row['definition'],
                        'division_name': row['division_name'],
                        'classification': classification
                    })
            
            elif self.classification_type == 'seo_divisions':
                # Get SEO divisions
                seo_divisions = anzsrc_df[
                    (anzsrc_df.index.get_level_values('classification') == 'SEO') & 
                    (anzsrc_df['level'] == 'division')
                ]
                
                for (classification, division_code, _), row in seo_divisions.iterrows():
                    division_code_str = str(division_code)  # Ensure string format
                    self.id_to_name[division_code_str] = row['division_name']
                    self.id_to_domain[division_code_str] = 'SEO'
                    # Store as dict for compatibility
                    self.classification_data.append({
                        'id': division_code_str,
                        'display_name': row['division_name'],
                        'definition': row['definition'],
                        'classification': classification
                    })
            
            elif self.classification_type == 'seo_groups':
                # Get SEO groups
                seo_groups = anzsrc_df[
                    (anzsrc_df.index.get_level_values('classification') == 'SEO') & 
                    (anzsrc_df['level'] == 'group')
                ]
                
                for (classification, division_code, group_code), row in seo_groups.iterrows():
                    group_code_str = str(group_code)  # Ensure string format
                    self.id_to_name[group_code_str] = row['group_name']
                    self.id_to_domain[group_code_str] = 'SEO'
                    self.id_to_field[group_code_str] = row['division_name']
                    # Store as dict for compatibility
                    self.classification_data.append({
                        'id': group_code_str,
                        'display_name': row['group_name'],
                        'definition': row['definition'],
                        'division_name': row['division_name'],
                        'classification': classification
                    })
                    
        except Exception as e:
            st.error(f"Error loading ANZSRC {self.classification_type} mappings: {e}")
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
    
    def get_classification_display_name(self) -> str:
        """Get a proper display name for the classification type"""
        display_names = {
            'domains': 'OpenAlex Domains',
            'fields': 'OpenAlex Fields',
            'subfields': 'OpenAlex Subfields',
            'topics': 'OpenAlex Topics',
            'for_divisions': 'ANZSRC FoR Divisions',
            'for_groups': 'ANZSRC FoR Groups',
            'seo_divisions': 'ANZSRC SEO Divisions',
            'seo_groups': 'ANZSRC SEO Groups'
        }
        return display_names.get(self.classification_type, self.classification_type.title() if self.classification_type else "Unknown")
    
    def get_classification_short_name(self) -> str:
        """Get a short name for the classification type (for labels)"""
        short_names = {
            'domains': 'Domain',
            'fields': 'Field',
            'subfields': 'Subfield',
            'topics': 'Topic',
            'for_divisions': 'FoR Division',
            'for_groups': 'FoR Group',
            'seo_divisions': 'SEO Division',
            'seo_groups': 'SEO Group'
        }
        return short_names.get(self.classification_type, self.classification_type.title() if self.classification_type else "Label")
    
    def get_hierarchical_level_name(self, level: str) -> str:
        """Get a proper display name for hierarchical levels"""
        anzsrc_types = ['for_divisions', 'for_groups', 'seo_divisions', 'seo_groups']
        
        if self.classification_type in anzsrc_types:
            if level == 'domain':
                return 'Classification System'  # FoR vs SEO
            elif level == 'field':
                return 'Division'  # ANZSRC division
            else:
                return level.title()
        else:
            # OpenAlex hierarchical levels
            return level.title()
            
    def _extract_classification(self, response_text: str) -> Optional[str]:
        """Extract classification ID(s) from model response"""
        if pd.isna(response_text):
            return None
        
        # Check if this is an ANZSRC classification (uses numeric codes) or OpenAlex (alphanumeric)
        anzsrc_types = ['for_divisions', 'for_groups', 'seo_divisions', 'seo_groups']
        
        if self.is_multiple_correct:
            # Handle multiple correct answers (comma-separated IDs)
            if self.classification_type in anzsrc_types:
                # ANZSRC uses numeric codes (e.g., "30,31", "3001,3002")
                match = re.search(r'ANSWER:\s*(["\']?)([\d,\s]+)\1', response_text)
                if match:
                    ids = [id_str.strip() for id_str in match.group(2).split(',') if id_str.strip()]
                    return ','.join(ids) if ids else None
            else:
                # OpenAlex uses alphanumeric codes (e.g., "T11881,T12345", "23,24")
                match = re.search(r'ANSWER:\s*(["\']?)([A-Za-z]?\d+(?:\s*,\s*[A-Za-z]?\d+)*)\1', response_text)
                if match:
                    ids = [id_str.strip() for id_str in match.group(2).split(',') if id_str.strip()]
                    return ','.join(ids) if ids else None
        else:
            # Handle single answer
            if self.classification_type in anzsrc_types:
                # ANZSRC uses numeric codes (e.g., "30", "3001")
                match = re.search(r'ANSWER:\s*(["\']?)(\d+)\1', response_text)
                return match.group(2) if match else None
            else:
                # OpenAlex uses alphanumeric codes (e.g., "T11881", "23", "2403")
                match = re.search(r'ANSWER:\s*(["\']?)([A-Za-z]?\d+)\1', response_text)
                return match.group(2) if match else None
        
        return None
    
    def _extract_reasoning(self, response_text: str) -> Optional[str]:
        """Extract reasoning from CoT model response (everything before ANSWER:)"""
        if pd.isna(response_text) or not self.is_cot:
            return None
        
        # Split on 'ANSWER:' and take everything before it as reasoning
        parts = response_text.split('ANSWER:')
        if len(parts) > 1:
            reasoning = parts[0].strip()
            return reasoning if reasoning else None
        
        return None
        
    def process_responses(self) -> None:
        """Process model responses and extract predictions"""
        if self.samples is None or self.messages is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        # Extract predictions from model responses
        model_responses = self.messages[self.messages.source == "generate"].copy()
        model_responses['predicted_id'] = model_responses['content'].apply(self._extract_classification)
        
        # Extract reasoning for CoT responses
        if self.is_cot:
            model_responses['reasoning'] = model_responses['content'].apply(self._extract_reasoning)
        
        # Merge with sample metadata
        merge_columns = ['sample_id', 'predicted_id']
        if self.is_cot:
            merge_columns.append('reasoning')
            
        self.analysis_df = self.samples.merge(
            model_responses[merge_columns], 
            on='sample_id', 
            how='left'
        )
        
        # Handle multiple predictions for multiple_correct setting
        if self.is_multiple_correct:
            # For multiple correct, we'll create separate rows for each prediction
            # but also keep the original comma-separated format for analysis
            self.analysis_df['predicted_ids_list'] = self.analysis_df['predicted_id'].apply(
                lambda x: x.split(',') if pd.notna(x) and ',' in str(x) else [x] if pd.notna(x) else []
            )
            self.analysis_df['num_predictions'] = self.analysis_df['predicted_ids_list'].apply(len)
        
        # Add predicted names, domains, and fields
        if self.is_multiple_correct:
            # For multiple answers, map each ID to its name
            self.analysis_df['predicted_names_list'] = self.analysis_df['predicted_ids_list'].apply(
                lambda ids: [self.id_to_name.get(id_str, id_str) for id_str in ids] if ids else []
            )
            # Join names for display
            self.analysis_df['predicted_name'] = self.analysis_df['predicted_names_list'].apply(
                lambda names: ', '.join(names) if names else None
            )
        else:
            # Single answer mapping
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
        
        if self.is_multiple_correct:
            # For multiple correct answers, count each individual prediction
            all_predictions = []
            for ids_list in self.analysis_df['predicted_ids_list']:
                if ids_list:
                    all_predictions.extend(ids_list)
            
            # Count occurrences and map to names if available
            if all_predictions:
                prediction_counts = pd.Series(all_predictions).value_counts()
                if self.id_to_name:
                    # Map IDs to names
                    prediction_counts.index = prediction_counts.index.map(
                        lambda x: self.id_to_name.get(x, x)
                    )
                return prediction_counts
            else:
                return pd.Series(dtype=object)
        else:
            # Single answer logic
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
        
        # Only include field distribution for group-level classifications (not domain/classification system)
        # This shows the division breakdown for ANZSRC groups or field breakdown for OpenAlex subfields/topics
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
        
        # Only include field funding distribution for group-level classifications
        # This shows the division breakdown for ANZSRC groups or field breakdown for OpenAlex subfields/topics
        if 'predicted_field' in self.analysis_df.columns:
            field_funding = self.analysis_df.groupby('predicted_field')['funding_amount'].sum().sort_values(ascending=False)
            distributions['field'] = field_funding
            
        return distributions
    
    def get_reasoning_data(self) -> Optional[pd.DataFrame]:
        """Get reasoning data for CoT responses"""
        if not self.is_cot or self.analysis_df is None:
            return None
        
        # Filter out rows without reasoning
        reasoning_df = self.analysis_df[self.analysis_df['reasoning'].notna()].copy()
        
        if len(reasoning_df) == 0:
            return None
        
        # Select relevant columns for reasoning display
        columns = ['metadata_grant_title', 'predicted_name', 'reasoning']
        if 'predicted_id' in reasoning_df.columns:
            columns.append('predicted_id')
        if 'metadata_grant_summary' in reasoning_df.columns:
            columns.append('metadata_grant_summary')
        
        return reasoning_df[columns].reset_index(drop=True)
    
    def get_evaluation_settings(self) -> Dict[str, any]:
        """Get evaluation settings information"""
        return {
            'template_type': self.template_type,
            'is_cot': self.is_cot,
            'is_multiple_correct': self.is_multiple_correct,
            'classification_type': self.classification_type
        }


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
    
    # Main layout with sidebar for metadata
    with st.sidebar:
        st.header("📊 Evaluation Overview")
        
        # File information
        file_date = datetime.datetime.fromtimestamp(file_info['file_modified'])
        classification_type_display = analyzer.get_classification_display_name()
        
        st.markdown(f"**File:** {file_info['file_name']}")
        st.markdown(f"**Classification Type:** {classification_type_display}")
        st.markdown(f"**Size:** {file_info['file_size_mb']:.1f} MB")
        st.markdown(f"**Modified:** {file_date.strftime('%Y-%m-%d %H:%M:%S')}")
        st.markdown(f"**Eval ID:** {model_info['eval_id'][:8]}...")
        
        st.markdown("---")
        
        # Basic metrics
        st.markdown("##### Quick Stats")
        st.metric("Total Samples", model_info['total_samples'])
        st.metric("Unique Labels Predicted", len(prediction_dist))
        st.metric("Most Common Label", prediction_dist.index[0])
        
        if model_info['total_time'] > 0:
            st.metric("Total Evaluation Time", f"{model_info['total_time']:.1f}s")
        
        # Add evaluation settings
        st.markdown("---")
        eval_settings = analyzer.get_evaluation_settings()
        st.markdown("##### Evaluation Settings")
        st.write(f"**Template Type:** {eval_settings['template_type']}")
        st.write(f"**Chain of Thought:** {'✅ Yes' if eval_settings['is_cot'] else '❌ No'}")
        st.write(f"**Multiple Correct:** {'✅ Yes' if eval_settings['is_multiple_correct'] else '❌ No'}")
        
        # Expandable detailed stats
        with st.expander("🔧 Detailed Technical Info"):
            st.markdown("##### Model & Token Usage")
            
            # Display model information
            if model_info['models_used']:
                st.write("**Models Used:**")
                for model in model_info['models_used']:
                    st.write(f"• {model}")
                
                # Token usage information
                for model, tokens in model_info['total_tokens'].items():
                    st.markdown(f"**{model} Token Usage:**")
                    st.write(f"• Input Tokens: {tokens['input_tokens']:,}")
                    st.write(f"• Output Tokens: {tokens['output_tokens']:,}")
                    st.write(f"• Total Tokens: {tokens['total_tokens']:,}")
                    if tokens['reasoning_tokens'] > 0:
                        st.write(f"• Reasoning Tokens: {tokens['reasoning_tokens']:,}")
                    
                    # Cost estimation (approximate for o4-mini)
                    if 'o4-mini' in model:
                        # Rough cost estimate: ~$0.50 per 1M tokens
                        estimated_cost = (tokens['total_tokens'] / 1_000_000) * 0.50
                        st.write(f"• Estimated Cost: ${estimated_cost:.3f}")
            
            st.markdown("##### Performance Metrics")
            total_tokens_all = sum(tokens['total_tokens'] for tokens in model_info['total_tokens'].values())
            st.write(f"• Total Tokens Used: {total_tokens_all:,}")
            if model_info['total_time'] > 0:
                st.write(f"• Avg Time per Sample: {model_info['avg_time_per_sample']:.2f}s")
                st.write(f"• Throughput: {model_info['total_samples']/model_info['total_time']:.1f} samples/sec")
            
            # Extract number of available labels from metadata if available
            num_labels = analyzer.samples['metadata_num_labels'].iloc[0] if 'metadata_num_labels' in analyzer.samples.columns else "Unknown"
            st.write(f"• Label Coverage: {len(prediction_dist)}/{num_labels}")
    
    # Main content area with tabs
    classification_display = analyzer.get_classification_short_name()
    
    # Create tabs for different visualizations
    tab_names = ["📊 Classification Results", "💰 Funding Analysis", "📋 Detailed Tables"]
    if analyzer.is_cot:
        tab_names.insert(-1, "🧠 Reasoning Analysis")  # Insert before the last tab
    
    tabs = st.tabs(tab_names)
    
    # Map tabs to variables
    tab1 = tabs[0]  # Classification Results
    tab2 = tabs[1]  # Funding Analysis
    if analyzer.is_cot:
        tab_reasoning = tabs[2]  # Reasoning Analysis
        tab3 = tabs[3]  # Detailed Tables
    else:
        tab3 = tabs[2]  # Detailed Tables
    
    with tab1:
        st.subheader(f"🎯 {classification_display} Distribution")
        
        # Add information about multiple predictions if applicable
        if analyzer.is_multiple_correct:
            if 'num_predictions' in analyzer.analysis_df.columns:
                avg_predictions = analyzer.analysis_df['num_predictions'].mean()
                st.info(f"📊 **Multiple Answer Mode**: On average, {avg_predictions:.1f} {classification_display.lower()}s predicted per grant")
        
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
            
            title_suffix = " (Individual Predictions)" if analyzer.is_multiple_correct else ""
            ax.set_title(f'Distribution of {classification_display}s in Dataset{title_suffix}', fontsize=16, fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            st.markdown(f"##### Top 5 {classification_display}s")
            top_5_df = pd.DataFrame({
                classification_display: prediction_dist.head(5).index,
                'Count': prediction_dist.head(5).values,
                'Percentage': (prediction_dist.head(5).values / prediction_dist.sum() * 100).round(1)
            })
            st.dataframe(top_5_df, use_container_width=True)
        
        # Show hierarchical distributions if available
        if hierarchical_dist:
            st.markdown("---")
            st.subheader("🏗️ Hierarchical Breakdown")
            
            # Create columns for hierarchical charts
            hierarchy_cols = st.columns(len(hierarchical_dist))
            
            for i, (level, distribution) in enumerate(hierarchical_dist.items()):
                with hierarchy_cols[i]:
                    level_display = analyzer.get_hierarchical_level_name(level)
                    st.markdown(f"##### By {level_display}")
                    
                    # Create a bar chart for hierarchical distribution
                    fig, ax = plt.subplots(figsize=(8, 6))
                    top_items = distribution.head(10)  # Show top 10
                    
                    bars = ax.barh(range(len(top_items)), top_items.values)
                    ax.set_yticks(range(len(top_items)))
                    ax.set_yticklabels(top_items.index)
                    ax.set_xlabel('Count')
                    ax.set_title(f'Top {level_display}s')
                    
                    # Add value labels on bars
                    for j, (bar, value) in enumerate(zip(bars, top_items.values)):
                        ax.text(value + 0.1, j, str(value), va='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
    
    with tab2:
        # Show funding-based distributions if funding data is available
        if len(funding_dist) > 0:
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
            
            # Show hierarchical funding distributions if available
            if hierarchical_funding_dist:
                st.markdown("---")
                st.subheader("🏗️ Hierarchical Funding Breakdown")
                
                # Create columns for hierarchical funding charts
                hierarchy_funding_cols = st.columns(len(hierarchical_funding_dist))
                
                for i, (level, distribution) in enumerate(hierarchical_funding_dist.items()):
                    with hierarchy_funding_cols[i]:
                        level_display = analyzer.get_hierarchical_level_name(level)
                        st.markdown(f"##### By {level_display}")
                        
                        # Create a bar chart for hierarchical funding distribution
                        fig, ax = plt.subplots(figsize=(8, 6))
                        top_items = distribution.head(10)  # Show top 10
                        
                        bars = ax.barh(range(len(top_items)), top_items.values)
                        ax.set_yticks(range(len(top_items)))
                        ax.set_yticklabels(top_items.index)
                        ax.set_xlabel('Total Funding ($)')
                        ax.set_title(f'Top {level_display}s by Funding')
                        
                        # Add value labels on bars
                        for j, (bar, value) in enumerate(zip(bars, top_items.values)):
                            ax.text(value + max(top_items.values) * 0.01, j, format_currency(value), va='center')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
        else:
            st.info("💡 No funding data available for this evaluation.")
    
    # Add reasoning tab if CoT is enabled
    if analyzer.is_cot:
        with tab_reasoning:
            st.subheader("🧠 Chain of Thought Reasoning Analysis")
            
            reasoning_data = analyzer.get_reasoning_data()
            
            if reasoning_data is not None and len(reasoning_data) > 0:
                st.markdown(f"Found reasoning for **{len(reasoning_data)}** samples")
                
                # Add search functionality
                search_term = st.text_input("🔍 Search reasoning (press Enter to search):", key="reasoning_search")
                
                # Filter data based on search
                display_data = reasoning_data
                if search_term:
                    mask = (
                        reasoning_data['reasoning'].str.contains(search_term, case=False, na=False) |
                        reasoning_data['metadata_grant_title'].str.contains(search_term, case=False, na=False)
                    )
                    display_data = reasoning_data[mask]
                    st.info(f"Showing {len(display_data)} of {len(reasoning_data)} samples matching '{search_term}'")
                
                # Display reasoning samples
                for idx, row in display_data.head(20).iterrows():  # Show first 20 results
                    with st.expander(f"📝 Grant: {row['metadata_grant_title'][:100]}... → Predicted: {row['predicted_name']}"):
                        
                        # Grant information
                        st.markdown("**Grant Title:**")
                        st.write(row['metadata_grant_title'])
                        
                        if 'metadata_grant_summary' in row and pd.notna(row['metadata_grant_summary']):
                            st.markdown("**Grant Summary:**")
                            st.write(row['metadata_grant_summary'][:500] + "..." if len(str(row['metadata_grant_summary'])) > 500 else row['metadata_grant_summary'])
                        
                        # Prediction information
                        st.markdown("**Predicted Classification:**")
                        predicted_text = row['predicted_name']
                        if 'predicted_id' in row and pd.notna(row['predicted_id']):
                            predicted_text += f" (ID: {row['predicted_id']})"
                        st.write(predicted_text)
                        
                        # Reasoning
                        st.markdown("**Model Reasoning:**")
                        st.write(row['reasoning'])
                
                if len(display_data) > 20:
                    st.info(f"Showing first 20 of {len(display_data)} results. Use search to filter for specific content.")
                    
            else:
                st.warning("⚠️ No reasoning data found. This could mean:")
                st.write("• The evaluation was not run with Chain of Thought (CoT) enabled")
                st.write("• The reasoning extraction failed")
                st.write("• The evaluation file doesn't contain complete reasoning data")
    
    with tab3:
        st.subheader("📋 Detailed Data Tables")
        
        # Multiple predictions summary if applicable
        if analyzer.is_multiple_correct and 'num_predictions' in analyzer.analysis_df.columns:
            st.markdown("##### Multiple Predictions Summary")
            pred_summary = analyzer.analysis_df['num_predictions'].value_counts().sort_index()
            pred_summary_df = pd.DataFrame({
                'Number of Predictions': pred_summary.index,
                'Number of Grants': pred_summary.values,
                'Percentage': (pred_summary.values / pred_summary.sum() * 100).round(2)
            })
            st.dataframe(pred_summary_df, use_container_width=True)
            st.markdown("---")
        
        # Classification counts table
        count_title = f"{classification_display} Counts"
        if analyzer.is_multiple_correct:
            count_title += " (Individual Predictions)"
        st.markdown(f"##### {count_title}")
        
        label_df = pd.DataFrame({
            classification_display: prediction_dist.index,
            'Count': prediction_dist.values,
            'Percentage': (prediction_dist.values / prediction_dist.sum() * 100).round(2)
        })
        st.dataframe(label_df, use_container_width=True, height=400)
        
        # Funding table if available
        if len(funding_dist) > 0:
            st.markdown("---")
            st.markdown(f"##### {classification_display} Funding Details")
            
            def format_currency(amount):
                if amount >= 1_000_000:
                    return f"${amount/1_000_000:.1f}M"
                elif amount >= 1_000:
                    return f"${amount/1_000:.0f}K"
                else:
                    return f"${amount:.0f}"
            
            funding_df = pd.DataFrame({
                classification_display: funding_dist.index,
                'Total Funding': [format_currency(x) for x in funding_dist.values],
                'Raw Amount': funding_dist.values,
                'Percentage': (funding_dist.values / funding_dist.sum() * 100).round(2)
            })
            st.dataframe(funding_df, use_container_width=True, height=400)


if __name__ == "__main__":
    main()
