import streamlit as st
import pandas as pd
from webutils import load_data, load_all_category_levels
from visualisation import create_category_hierarchy_visualization, create_category_distribution_chart
import config

st.set_page_config(page_title="Category Hierarchy", page_icon="🏷️", layout="wide")

st.markdown("# 🏷️ Category Hierarchy")
st.sidebar.header("Category Hierarchy")
st.markdown("Explore the abstracted research categories generated through hierarchical clustering of keywords.")

# Load data
keywords, grants, categories = load_data()

if keywords is None or grants is None or categories is None:
    st.error("Failed to load data. Please check your data files.")
    st.stop()

# Load available category levels
available_levels = load_all_category_levels()

if not available_levels:
    st.error("No category levels found. Please run the categorization process first.")
    st.stop()

# Sidebar controls
with st.sidebar:
    st.subheader("Category Settings")
    
    # Level selection
    selected_level = st.selectbox(
        "Category Abstraction Level",
        options=available_levels,
        index=len(available_levels) - 1 if available_levels else 0,
        help="Higher levels represent more abstracted/generalized categories"
    )
    
    # Visualization type
    viz_type = st.selectbox(
        "Visualization Type",
        options=["Hierarchy Treemap", "Distribution Bar Chart", "Distribution Pie Chart", "Horizontal Bar Chart"],
        index=0,
        help="Choose how to visualize the category data"
    )
    
    # Display options
    st.subheader("Display Options")
    
    show_descriptions = st.checkbox(
        "Show Category Descriptions",
        value=True,
        help="Display detailed descriptions of each category"
    )
    
    # Add display limit controls
    max_categories_display = st.slider(
        "Max Categories to Display",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="Limit number of categories shown to prevent performance issues"
    )
    
    if viz_type == "Hierarchy Treemap":
        treemap_height = st.slider(
            "Visualization Height",
            min_value=400,
            max_value=1200,
            value=800,
            step=50,
            help="Height of the treemap visualization"
        )
        
        treemap_font_size = st.slider(
            "Font Size",
            min_value=8,
            max_value=16,
            value=12,
            help="Font size for treemap labels"
        )
        
        max_keywords = st.slider(
            "Max Keywords per Category",
            min_value=10,
            max_value=100,
            value=50,
            step=5,
            help="Maximum number of keywords to show per category"
        )

# Load category data for the selected level
try:
    categories_data = config.Categories.load(selected_level)
    
    # Handle mapping data - only load if not at level 0 and mapping exists
    if selected_level > 0:
        # Check if mapping exists before attempting to load
        if config.Categories.has_mapping(selected_level):
            keyword_mapping_df = config.Categories.load_mapping(selected_level)
            
            # Check if mapping data is valid
            if keyword_mapping_df is None or (hasattr(keyword_mapping_df, 'empty') and keyword_mapping_df.empty):
                st.warning(f"Mapping file exists but is empty for level {selected_level}")
                keyword_mapping = {}
            else:
                # Convert DataFrame to dictionary
                keyword_mapping = dict(zip(keyword_mapping_df['source_item'], keyword_mapping_df['target_item']))
        else:
            source_level = selected_level - 1 if selected_level > 0 else 0
            st.warning(f"No mapping file found for level {source_level} → {selected_level}. Run the categorization process to generate mappings.")
            keyword_mapping = {}
    else:
        # For level 0, there's no mapping (it's the base level)
        keyword_mapping = {}
    
    if categories_data.empty:
        st.error(f"No category data found for level {selected_level}")
        st.stop()
        
except Exception as e:
    st.error(f"Error loading category data: {e}")
    st.stop()

# Main content area
st.subheader(f"Category Analysis - Level {selected_level}")

# Display basic statistics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Categories", len(categories_data))
with col2:
    source_level = selected_level - 1 if selected_level > 0 else 0
    st.metric(f"Mapped Level {source_level} Items", len(keyword_mapping))
with col3:
    mapped_items = len([k for k in keyword_mapping.values() if k in categories_data['name'].values])
    st.metric("Successfully Mapped", mapped_items)

# Create visualization based on selected type
try:
    if viz_type == "Hierarchy Treemap":
        if keyword_mapping:
            fig = create_category_hierarchy_visualization(
                categories_data, 
                keyword_mapping,
                height=treemap_height,
                font_size=treemap_font_size,
                max_keywords_per_category=max_keywords
            )
            st.plotly_chart(fig, width='stretch')
        else:
            if selected_level == 0:
                st.info("Treemap visualization is not available for level 0 (base keywords). Please select a higher abstraction level.")
            else:
                st.warning("No keyword mapping available for treemap visualization")
            
    elif viz_type == "Distribution Bar Chart":
        if keyword_mapping:
            fig = create_category_distribution_chart(
                keyword_mapping,
                chart_type="bar"
            )
            st.plotly_chart(fig, width='stretch')
        else:
            if selected_level == 0:
                st.info("Distribution charts require mapping data. Please select a higher abstraction level.")
            else:
                st.warning("No keyword mapping available for distribution chart")
            
    elif viz_type == "Distribution Pie Chart":
        if keyword_mapping:
            fig = create_category_distribution_chart(
                keyword_mapping,
                chart_type="pie"
            )
            st.plotly_chart(fig, width='stretch')
        else:
            if selected_level == 0:
                st.info("Distribution charts require mapping data. Please select a higher abstraction level.")
            else:
                st.warning("No keyword mapping available for pie chart")
            
    elif viz_type == "Horizontal Bar Chart":
        if keyword_mapping:
            fig = create_category_distribution_chart(
                keyword_mapping,
                chart_type="horizontal_bar"
            )
            st.plotly_chart(fig, width='stretch')
        else:
            if selected_level == 0:
                st.info("Distribution charts require mapping data. Please select a higher abstraction level.")
            else:
                st.warning("No keyword mapping available for horizontal bar chart")
            
except Exception as e:
    st.error(f"Error creating visualization: {e}")

# Display category details
if show_descriptions and not categories_data.empty:
    st.subheader("Category Details")
    
    # Apply display limit
    total_categories = len(categories_data)
    if total_categories > max_categories_display:
        st.info(f"Showing {max_categories_display} out of {total_categories} categories. Adjust the display limit in the sidebar to see more.")
    
    # Calculate keyword counts for each category
    category_keyword_counts = {}
    if keyword_mapping:
        for keyword, category in keyword_mapping.items():
            category_keyword_counts[category] = category_keyword_counts.get(category, 0) + 1
    
    # Sort categories by keyword count (descending) and apply limit
    sorted_categories = categories_data.copy()
    if category_keyword_counts:
        sorted_categories['keyword_count'] = sorted_categories['name'].map(lambda x: category_keyword_counts.get(x, 0))
        sorted_categories = sorted_categories.sort_values('keyword_count', ascending=False)
    
    # Apply display limit
    display_categories = sorted_categories.head(max_categories_display)
    
    # Prepare data for dataframe display
    source_level = selected_level - 1 if selected_level > 0 else 0
    item_type = "keywords" if source_level == 0 else f"level {source_level} items"
    
    # Create dataframe with category information
    category_display_data = []
    for idx, row in display_categories.iterrows():
        category_name = row['name']
        description = row['description']
        item_count = category_keyword_counts.get(category_name, 0)
        
        # Get sample items for this category
        sample_items = ""
        if keyword_mapping and item_count > 0:
            sample_items_list = [k for k, v in keyword_mapping.items() if v == category_name][:5]
            if sample_items_list:
                sample_items = ", ".join(sample_items_list)
                if len(sample_items_list) == 5 and item_count > 5:
                    sample_items += f" (+ {item_count - 5} more)"
        
        category_display_data.append({
            "Category": category_name,
            f"{item_type.title()} Count": item_count,
            "Description": description,
            f"Sample {item_type.title()}": sample_items if sample_items else "No mapping available"
        })
    
    # Display as dataframe
    if category_display_data:
        categories_df = pd.DataFrame(category_display_data)
        
        # Configure columns
        column_config = {
            "Category": st.column_config.TextColumn(
                "Category Name",
                help="Name of the research category",
                max_chars=50
            ),
            f"{item_type.title()} Count": st.column_config.NumberColumn(
                f"{item_type.title()} Count",
                help=f"Number of {item_type} mapped to this category",
                format="%d"
            ),
            "Description": st.column_config.TextColumn(
                "Description",
                help="Detailed description of the category",
                max_chars=200
            ),
            f"Sample {item_type.title()}": st.column_config.TextColumn(
                f"Sample {item_type.title()}",
                help=f"Sample {item_type} belonging to this category",
                max_chars=150
            )
        }
        
        st.dataframe(
            categories_df,
            column_config=column_config,
            hide_index=True,
            use_container_width=True,
            height=400
        )
    else:
        st.info("No categories to display with current filters.")

# Display mapping statistics
if selected_level > 0 and keyword_mapping:
    source_level = selected_level - 1 if selected_level > 0 else 0
    item_type = "Keywords" if source_level == 0 else f"Level {source_level} Items"
    
    st.subheader(f"{item_type} Mapping Statistics")
    
    # Get source data for comparison
    if source_level == 0:
        all_source_names = set(keywords['name'].tolist())
    else:
        source_data = config.Categories.load(source_level)
        all_source_names = set(source_data['name'].tolist())
    
    mapped_source_names = set(keyword_mapping.keys())
    unmapped_items = all_source_names - mapped_source_names
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(f"Total {item_type}", len(all_source_names))
        st.metric(f"Mapped {item_type}", len(mapped_source_names))
    with col2:
        mapping_rate = len(mapped_source_names) / len(all_source_names) * 100 if all_source_names else 0
        st.metric("Mapping Rate", f"{mapping_rate:.1f}%")
        st.metric(f"Unmapped {item_type}", len(unmapped_items))
    
    if unmapped_items and st.checkbox(f"Show unmapped {item_type.lower()}"):
        st.write(f"**Unmapped {item_type}:**")
        unmapped_list = sorted(list(unmapped_items))
        st.write(", ".join(unmapped_list[:50]))
        if len(unmapped_list) > 50:
            st.write(f"... and {len(unmapped_list) - 50} more")
elif selected_level == 0:
    st.subheader("Base Level Information")
    st.info("Level 0 represents the base keywords extracted from research data. No mapping statistics are available for this level.")

# Display level progression information
if len(available_levels) > 1:
    st.subheader("Abstraction Level Information")
    
    # Get available mappings
    available_mappings = config.Categories.get_available_mappings()
    mapping_dict = {target: source for source, target in available_mappings}
    
    level_info = []
    for level in available_levels:
        try:
            # Ensure level is an integer
            level = int(level)
            level_categories = config.Categories.load(level)
            
            # Check if mapping exists and get count
            if level > 0 and level in mapping_dict:
                level_mapping_df = config.Categories.load_mapping(level)
                level_mapping_count = len(level_mapping_df) if not level_mapping_df.empty else 0
                mapping_status = "✅ Available"
            elif level > 0:
                level_mapping_count = 0
                mapping_status = "❌ Missing"
            else:
                level_mapping_count = 0
                mapping_status = "N/A (Base level)"
            
            # Determine source level for better description
            source_level = level - 1 if level > 0 else 0
            source_description = "Keywords" if source_level == 0 else f"Level {source_level}"
            
            level_info.append({
                "Level": level,
                "Items": len(level_categories),
                f"Mapped from {source_description}": level_mapping_count if level > 0 else "N/A",
                "Mapping Status": mapping_status,
                "Description": f"Level {level} abstraction" + (" (Base keywords)" if level == 0 else "")
            })
        except Exception as e:
            st.warning(f"Error processing level {level}: {e}")
            continue
    
    if level_info:
        level_df = pd.DataFrame(level_info)
        st.dataframe(level_df, use_container_width=True)
        
        # Show available mappings summary
        if available_mappings:
            st.write("**Available Mappings:**")
            mapping_summary = []
            for source, target in available_mappings:
                source_desc = "Keywords" if source == 0 else f"Level {source}"
                mapping_summary.append(f"• {source_desc} → Level {target}")
            st.write("\n".join(mapping_summary))
        
        st.info("""
        **Level Guide:**
        - Level 0: Base keywords extracted from research data
        - Level 1+: Increasingly abstracted category levels
        - Higher levels represent broader, more generalized themes
        - Each level can be mapped from any lower level
        - ✅ = Mapping file exists, ❌ = Mapping needs to be generated
        """)
else:
    st.subheader("Mapping Information")
    available_mappings = config.Categories.get_available_mappings()
    if available_mappings:
        st.write("**Available Mappings:**")
        for source, target in available_mappings:
            source_desc = "Keywords" if source == 0 else f"Level {source}"
            st.write(f"• {source_desc} → Level {target}")
    else:
        st.info("No mappings have been generated yet. Run the categorization process to create level mappings.")
