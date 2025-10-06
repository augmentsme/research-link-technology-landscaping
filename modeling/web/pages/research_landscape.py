"""
Research Landscape Page
Explore the hierarchical structure of research categories and keywords through interactive treemap visualization.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# Add the web directory to Python path to import shared_utils
web_dir = str(Path(__file__).parent.parent)
if web_dir not in sys.path:
    sys.path.insert(0, web_dir)

from visualisation import create_research_landscape_treemap
from shared_utils import (
    load_data,
)


@dataclass
class TreemapConfig:
    """Configuration for treemap visualization"""
    max_research_fields: Optional[int]
    max_categories_per_field: Optional[int]
    max_keywords_per_category: Optional[int]
    treemap_height: int
    font_size: int


class TreemapVisualizer:
    """Manages the research landscape treemap visualization"""
    
    def __init__(self, categories_df: pd.DataFrame):
        self.categories_df = categories_df
        self.categories_list = categories_df.to_dict('records')
    
    def render_visualization(self, config: TreemapConfig):
        """Render the treemap visualization with given configuration"""
        with st.spinner("Creating research landscape treemap..."):
            fig_treemap = create_research_landscape_treemap(
                categories=self.categories_list,
                classification_results=[],  # Empty for now
                title="Research Landscape: Research Fields → Categories → Keywords",
                height=config.treemap_height,
                font_size=config.font_size,
                max_research_fields=config.max_research_fields,
                max_categories_per_field=config.max_categories_per_field,
                max_keywords_per_category=config.max_keywords_per_category
            )
            
            if fig_treemap is not None:
                st.plotly_chart(fig_treemap, use_container_width=True)
                
                # Show statistics
                self._show_statistics()
                
                # Debug expander showing underlying data
                self._show_debug_data()
            else:
                st.warning("No data available for the selected parameters.")
    
    def _show_statistics(self):
        """Display statistics about the research landscape"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Categories", len(self.categories_df))
        with col2:
            total_keywords = sum(len(cat.get('keywords', [])) for cat in self.categories_list)
            st.metric("Total Keywords", total_keywords)
        with col3:
            unique_fields = len(set(cat.get('field_of_research', cat.get('for_code', 'Unknown')) for cat in self.categories_list))
            st.metric("Unique Research Fields", unique_fields)
    
    def _show_debug_data(self):
        """Show debug data in an expander"""
        with st.expander("🔍 Debug: View Underlying Data", expanded=False):
            st.subheader("Categories DataFrame")
            st.write(f"**Shape:** {self.categories_df.shape}")
            st.dataframe(self.categories_df, use_container_width=True)
            
            st.subheader("Categories List (Dict Format)")
            st.write(f"**Length:** {len(self.categories_list)} categories")
            # Show first few categories as example
            st.json(self.categories_list[:3] if len(self.categories_list) >= 3 else self.categories_list)


class SidebarControls:
    """Handles all sidebar UI controls"""
    
    def __init__(self):
        pass
    
    def render_sidebar(self) -> tuple[TreemapConfig, bool]:
        """Render all sidebar controls and return configuration"""
        st.sidebar.empty()
        
        with st.sidebar:
            st.subheader("🌳 Treemap Settings")
            
            config = self._render_treemap_settings()
            update_settings = self._render_update_button()
            
            return config, update_settings
    
    def _render_treemap_settings(self) -> TreemapConfig:
        """Render treemap configuration controls"""
        # Filtering Options in Expander
        with st.expander("🔍 Filtering Options", expanded=True):
            max_research_fields = st.selectbox(
                "Maximum Research Fields",
                options=[None, 5, 10, 15, 20],
                index=2,
                format_func=lambda x: "All" if x is None else str(x),
                help="Maximum number of research fields to include",
                key="landscape_max_research_fields"
            )
            
            max_categories_per_field = st.selectbox(
                "Maximum categories per research field",
                options=[None, 3, 5, 10, 15],
                index=2,
                format_func=lambda x: "All" if x is None else str(x),
                help="Maximum number of categories to show per research field",
                key="landscape_max_categories_per_field"
            )
            
            max_keywords_per_category = st.selectbox(
                "Maximum keywords per category",
                options=[None, 5, 10, 20, 30],
                index=2,
                format_func=lambda x: "All" if x is None else str(x),
                help="Maximum number of keywords to show per category",
                key="landscape_max_keywords_per_category"
            )
        
        # Display Settings in Expander
        with st.expander("⚙️ Display Settings", expanded=False):
            treemap_height = st.slider(
                "Visualization height",
                min_value=400,
                max_value=1200,
                value=800,
                step=50,
                key="landscape_treemap_height"
            )
            
            font_size = st.slider(
                "Font size",
                min_value=6,
                max_value=16,
                value=9,
                key="landscape_font_size"
            )
        
        return TreemapConfig(
            max_research_fields=max_research_fields,
            max_categories_per_field=max_categories_per_field,
            max_keywords_per_category=max_keywords_per_category,
            treemap_height=treemap_height,
            font_size=font_size
        )
    
    def _render_update_button(self) -> bool:
        """Render update settings button"""
        st.markdown("---")
        return st.button("Update Settings", type="primary", use_container_width=True, key="landscape_update_button")


class ResearchLandscapePage:
    """Main page class that orchestrates all components"""
    
    def __init__(self):
        self.keywords_df = None
        self.grants_df = None
        self.categories_df = None
        self.setup_page()
    
    def setup_page(self):

        
        st.header("Research Landscape Treemap")
        st.markdown("Explore the hierarchical structure of research categories and keywords.")
        
        self._load_data()
        
        if self.categories_df is None:
            st.error("Unable to load categories data. Please check your data files.")
            return
    
    def _load_data(self):
        """Load and store data"""
        self.keywords_df, self.grants_df, self.categories_df = load_data()
    
    def run(self):
        """Main execution method"""
        # Create sidebar controls
        sidebar_controls = SidebarControls()
        
        # Get configuration from sidebar
        treemap_config, update_settings = sidebar_controls.render_sidebar()
        
        # Check if visualization should be generated
        should_generate = self._should_generate_visualization(update_settings)
        
        if should_generate:
            # Create and render treemap visualization
            visualizer = TreemapVisualizer(self.categories_df)
            visualizer.render_visualization(treemap_config)
        else:
            st.info("💡 Adjust settings in the sidebar and click 'Update Settings' to generate a new visualization.")
    
    def _should_generate_visualization(self, update_settings: bool) -> bool:
        """Determine if visualization should be generated"""
        # Use session state to track if this is the first load
        if "landscape_initialized" not in st.session_state:
            st.session_state.landscape_initialized = True
            return True
        elif update_settings:
            return True
        else:
            return False


def main():
    """Main function to run the research landscape page"""
    landscape_page = ResearchLandscapePage()
    landscape_page.run()


if __name__ == "__main__":
    main()
