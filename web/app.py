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
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import yaml

# Configure matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    try:
        # Handle relative paths by making them relative to the project root
        if not Path(config_path).is_absolute():
            # Try to find config relative to this file's location
            current_dir = Path(__file__).parent.parent
            config_path = str(current_dir / config_path)
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"Configuration file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        st.error(f"Error parsing YAML configuration: {e}")
        return {}


class KeywordWorkflowAnalyzer:
    """Analyzer for the keyword harmonisation workflow"""
    
    def __init__(self, config: Dict):
        # Get paths from configuration
        root_dir = config.get('root_dir', '.')
        data_dirname = config.get('data_dirname', 'data')
        
        self.base_data_dir = Path(root_dir) / data_dirname
        self.combined_results_file = self.base_data_dir / "combined_workflow_results.json"
        
        # Data storage
        self.combined_data: Optional[Dict] = None

    def load_combined_data(self) -> bool:
        """Load combined workflow results from JSON file"""
        if not self.combined_results_file.exists():
            st.warning(f"Combined results file not found: {self.combined_results_file}")
            return False
            
        try:
            with open(self.combined_results_file, 'r') as f:
                self.combined_data = json.load(f)
            return True
        except Exception as e:
            st.error(f"Error loading combined results: {e}")
            return False

    def show_keyword_extractions_tab(self):
        """Show keyword extractions for grants"""
        st.header("🔤 Keywords Extracted for Grants")
        
        if not self.combined_data:
            st.error("No combined workflow data available")
            return
        
        grant_extractions = self.combined_data.get('grant_extractions', {})
        
        if not grant_extractions:
            st.warning("No keyword extraction data found")
            return
        
        # Display each grant and its extracted keywords
        for grant_id, keywords in grant_extractions.items():
            with st.expander(f"Grant: {grant_id} ({len(keywords)} keywords)"):
                if keywords:
                    st.write(", ".join(keywords))
                else:
                    st.write("No keywords extracted")

    def show_keyword_harmonisation_tab(self):
        """Show how keywords are harmonised"""
        st.header("🔗 Keywords Harmonisation")
        
        if not self.combined_data:
            st.error("No combined workflow data available")
            return
        
        harmonisation = self.combined_data.get('harmonisation', {})
        
        if not harmonisation:
            st.warning("No keyword harmonisation data found")
            return
        
        # Show harmonised keywords
        harmonised_keywords = harmonisation.get('harmonised_keywords', [])
        if harmonised_keywords:
            st.subheader("Harmonised Keywords:")
            st.write(", ".join(harmonised_keywords))
        
        # Show keyword mappings (original -> harmonised)
        mappings = harmonisation.get('keyword_mappings', {})
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
        merged_groups = harmonisation.get('merged_groups', {})
        if merged_groups:
            st.subheader("Merged Groups:")
            for harmonised_keyword, original_keywords in merged_groups.items():
                if len(original_keywords) > 1:
                    st.write(f"**{harmonised_keyword}** ← {', '.join(original_keywords)}")

    def show_keyword_assignments_tab(self):
        """Show how harmonised keywords link to grants"""
        st.header("📋 Harmonised Keywords Linked to Grants")
        
        if not self.combined_data:
            st.error("No combined workflow data available")
            return
        
        assignments = self.combined_data.get('assignments', [])
        
        if not assignments:
            st.warning("No keyword assignment data found")
            return
        
        # Display each grant and its assigned harmonised keywords
        for assignment in assignments:
            grant_id = assignment.get('grant_id', 'Unknown')
            grant_title = assignment.get('grant_title', 'No title')
            keywords = assignment.get('harmonised_keywords', [])
            
            with st.expander(f"Grant: {grant_id} - {grant_title[:100]}... ({len(keywords)} keywords)"):
                if keywords:
                    st.write(", ".join(keywords))
                else:
                    st.write("No harmonised keywords assigned")

def main():
    # Load configuration
    config = load_config("config.yaml")
    web_config = config.get('web', {})
    
    st.set_page_config(
        page_title="Keyword Analysis Dashboard",
        page_icon=web_config.get('icon', "🔍"),
        layout=web_config.get('layout', "wide")
    )
    
    st.title(f"{web_config.get('icon', '🔍')} {web_config.get('title', 'Research Grant Keyword Analysis Dashboard')}")
    
    # Initialize analyzer with configuration
    analyzer = KeywordWorkflowAnalyzer(config)
    
    # Load combined data
    with st.spinner("Loading workflow results..."):
        if not analyzer.load_combined_data():
            st.error("Failed to load combined workflow results. Please ensure the data has been generated.")
            st.info("Run `python modeling/keyword_assignment.py` to generate the combined results file.")
            return
    
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
