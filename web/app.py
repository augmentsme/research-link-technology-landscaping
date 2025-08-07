"""
Streamlit Dashboard for Research Grant Keyword Analysis Workflow

This application visualizes the keyword harmonisation workflow:
1. Grant Data Overview
2. Keyword Extraction Results
3. Keyword Harmonisation Analysis
4. Grant Keyword Assignments
"""

import json
from pathlib import Path
from typing import Dict, Optional, List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
from inspect_ai.log import read_eval_log

# Configure matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")


class KeywordWorkflowAnalyzer:
    """Analyzer for the keyword harmonisation workflow"""
    
    def __init__(self, 
                 base_data_dir: str = "/home/lcheng/oz318/research-link-technology-landscaping/data",
                 logs_dir: str = "/home/lcheng/oz318/research-link-technology-landscaping/logs"):
        self.base_data_dir = Path(base_data_dir)
        self.logs_dir = Path(logs_dir)
        
        # Data storage
        self.grants_data: Optional[pd.DataFrame] = None
        self.keyword_extractions: Optional[pd.DataFrame] = None
        self.harmonised_keywords: Dict = {}
        self.keyword_assignments: Optional[pd.DataFrame] = None
        self.assignment_summary: Dict = {}

    def load_keyword_extractions(self) -> bool:
        """Load keyword extraction results from evaluation logs"""
        pattern = "*extract*keywords*.eval"
        eval_files = list(self.logs_dir.glob(pattern))
        
        if not eval_files:
            st.warning("No keyword extraction evaluation files found")
            return False
            
        # Use the most recent file
        latest_file = max(eval_files, key=lambda x: x.stat().st_mtime)
        
        try:
            log = read_eval_log(str(latest_file))
            extraction_data = []
            
            for sample in log.samples:
                if sample.output and sample.output.completion:
                    try:
                        keywords_result = json.loads(sample.output.completion)
                        
                        # Collect all keywords
                        all_keywords = []
                        for category in ['keywords', 'methodology_keywords', 'application_keywords', 'technology_keywords']:
                            if category in keywords_result:
                                keywords = keywords_result[category]
                                all_keywords.extend(keywords)
                        
                        extraction_data.append({
                            'grant_id': sample.id or f"grant_{len(extraction_data)}",
                            'keywords': all_keywords
                        })
                        
                    except json.JSONDecodeError:
                        continue
            
            if extraction_data:
                self.keyword_extractions = pd.DataFrame(extraction_data)
                return True
            else:
                return False
                
        except Exception as e:
            st.error(f"Error loading keyword extractions: {e}")
            return False

    def load_harmonised_keywords(self) -> bool:
        """Load keyword harmonisation results from evaluation logs"""
        pattern = "*harmon*keywords*.eval"
        eval_files = list(self.logs_dir.glob(pattern))
        
        if not eval_files:
            st.warning("No keyword harmonisation evaluation files found")
            return False
            
        # Use the most recent file
        latest_file = max(eval_files, key=lambda x: x.stat().st_mtime)
        
        try:
            log = read_eval_log(str(latest_file))
            
            for sample in log.samples:
                if sample.output and sample.output.completion:
                    try:
                        harmonisation_result = json.loads(sample.output.completion)
                        
                        if 'harmonised_keywords' in harmonisation_result:
                            self.harmonised_keywords = {
                                'harmonised_keywords': harmonisation_result.get('harmonised_keywords', []),
                                'keyword_mappings': harmonisation_result.get('keyword_mappings', {}),
                                'merged_groups': harmonisation_result.get('merged_groups', {}),
                                'unchanged_keywords': harmonisation_result.get('unchanged_keywords', [])
                            }
                            return True

                    except json.JSONDecodeError:
                        continue
            
            return False
            
        except Exception as e:
            st.error(f"Error loading harmonised keywords: {e}")
            return False

    def load_keyword_assignments(self) -> bool:
        """Load grant keyword assignment results"""
        assignment_file = "grant_keyword_assignments.json"
        assignment_path = self.base_data_dir / assignment_file
        
        if not assignment_path.exists():
            st.warning(f"Keyword assignment file not found: {assignment_path}")
            return False
            
        try:
            with open(assignment_path, 'r') as f:
                data = json.load(f)
            
            assignments = data.get('assignments', [])
            assignment_data = []
            
            for assignment in assignments:
                assignment_data.append({
                    'grant_id': assignment['grant_id'],
                    'grant_title': assignment['grant_title'],
                    'harmonised_keywords': assignment.get('harmonised_keywords', [])
                })
            
            if assignment_data:
                self.keyword_assignments = pd.DataFrame(assignment_data)
                return True
            else:
                return False
                
        except Exception as e:
            st.error(f"Error loading keyword assignments: {e}")
            return False

    def show_keyword_extractions_tab(self):
        """Show keyword extractions for grants"""
        st.header("� Keywords Extracted for Grants")
        
        if self.keyword_extractions is None:
            st.error("No keyword extraction data available")
            return
        
        # Display each grant and its extracted keywords
        for _, row in self.keyword_extractions.iterrows():
            grant_id = row['grant_id']
            keywords = row['keywords']
            
            with st.expander(f"Grant: {grant_id} ({len(keywords)} keywords)"):
                if keywords:
                    st.write(", ".join(keywords))
                else:
                    st.write("No keywords extracted")

    def show_keyword_harmonisation_tab(self):
        """Show how keywords are harmonised"""
        st.header("🔗 Keywords Harmonisation")
        
        if not self.harmonised_keywords:
            st.error("No keyword harmonisation data available")
            return
        
        # Show harmonised keywords
        harmonised = self.harmonised_keywords.get('harmonised_keywords', [])
        if harmonised:
            st.subheader("Harmonised Keywords:")
            st.write(", ".join(harmonised))
        
        # Show keyword mappings (original -> harmonised)
        mappings = self.harmonised_keywords.get('keyword_mappings', {})
        if mappings:
            st.subheader("Keyword Mappings (Original → Harmonised):")
            mapping_data = []
            for original, harmonised in mappings.items():
                mapping_data.append({
                    'Original Keyword': original,
                    'Harmonised Keyword': harmonised
                })
            
            if mapping_data:
                df = pd.DataFrame(mapping_data)
                st.dataframe(df, use_container_width=True)
        
        # Show merged groups
        merged_groups = self.harmonised_keywords.get('merged_groups', {})
        if merged_groups:
            st.subheader("Merged Groups:")
            for harmonised_keyword, original_keywords in merged_groups.items():
                if len(original_keywords) > 1:
                    st.write(f"**{harmonised_keyword}** ← {', '.join(original_keywords)}")

    def show_keyword_assignments_tab(self):
        """Show how harmonised keywords link to grants"""
        st.header("📋 Harmonised Keywords Linked to Grants")
        
        if self.keyword_assignments is None:
            st.error("No keyword assignment data available")
            return
        
        # Display each grant and its assigned harmonised keywords
        for _, row in self.keyword_assignments.iterrows():
            grant_id = row['grant_id']
            grant_title = row['grant_title']
            keywords = row['harmonised_keywords']
            
            with st.expander(f"Grant: {grant_id} - {grant_title[:100]}... ({len(keywords)} keywords)"):
                if keywords:
                    st.write(", ".join(keywords))
                else:
                    st.write("No harmonised keywords assigned")

def main():
    st.set_page_config(
        page_title="Keyword Analysis Dashboard",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 Research Grant Keyword Analysis Dashboard")
    
    # Initialize analyzer
    analyzer = KeywordWorkflowAnalyzer()
    
    # Load data
    with st.spinner("Loading data..."):
        extractions_loaded = analyzer.load_keyword_extractions()
        harmonisation_loaded = analyzer.load_harmonised_keywords()
        assignments_loaded = analyzer.load_keyword_assignments()
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "🔤 Keywords Extracted",
        "🔗 Keywords Harmonised", 
        "📋 Keywords Assigned to Grants"
    ])
    
    with tab1:
        analyzer.show_keyword_extractions_tab()
    
    with tab2:
        analyzer.show_keyword_harmonisation_tab()
    
    with tab3:
        analyzer.show_keyword_assignments_tab()

if __name__ == "__main__":
    main()
