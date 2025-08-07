"""
Streamlit Dashboard for Research Grant Analysis Workflow

This application visualizes all steps of the research grant analysis workflow:
1. Grant Data Overview
2. Keyword Extraction Results
3. Keyword Clustering Analysis
4. Topic Assignment Results
"""

import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
from inspect_ai.log import read_eval_log

# Configure matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")


class WorkflowAnalyzer:
    """Complete analyzer for the research grant analysis workflow"""
    
    def __init__(self, 
                 base_data_dir: str = "/home/lcheng/oz318/research-link-technology-landscaping/data",
                 logs_dir: str = "/home/lcheng/oz318/research-link-technology-landscaping/logs"):
        self.base_data_dir = Path(base_data_dir)
        self.logs_dir = Path(logs_dir)
        
        # Data storage
        self.grants_data: Optional[pd.DataFrame] = None
        self.keyword_extractions: Optional[pd.DataFrame] = None
        self.keyword_clusters: Dict = {}
        self.topic_assignments: Optional[pd.DataFrame] = None
        self.assignment_summary: Dict = {}
        
    def load_grants_data(self, grants_file: str = "active_grants.json") -> bool:
        """Load the original grants data"""
        grants_path = self.base_data_dir / grants_file
        
        if not grants_path.exists():
            st.error(f"Grants file not found: {grants_path}")
            return False
        
        try:
            with open(grants_path, 'r', encoding='utf-8') as f:
                grants_data = json.load(f)
            
            # Convert to DataFrame
            self.grants_data = pd.DataFrame(grants_data)
            
            # Clean and process data
            self.grants_data['has_summary'] = self.grants_data['grant_summary'].notna() & (self.grants_data['grant_summary'].str.len() > 0)
            self.grants_data['funding_amount'] = pd.to_numeric(self.grants_data['funding_amount'], errors='coerce')
            self.grants_data['summary_length'] = self.grants_data['grant_summary'].str.len().fillna(0)
            
            return True
            
        except Exception as e:
            st.error(f"Error loading grants data: {e}")
            return False
    
    def load_keyword_extractions(self, pattern: str = "*extract*keywords*.eval") -> bool:
        """Load keyword extraction results from evaluation logs"""
        eval_files = list(self.logs_dir.glob(pattern))
        
        if not eval_files:
            st.warning("No keyword extraction evaluation files found")
            return False
        
        # Use the most recent file
        eval_file = max(eval_files, key=lambda x: x.stat().st_mtime)
        
        try:
            log = read_eval_log(str(eval_file))
            
            extraction_data = []
            for sample in log.samples:
                if sample.output and sample.output.completion:
                    try:
                        keywords_result = json.loads(sample.output.completion)
                        grant_id = sample.metadata.get('grant_id', '') if sample.metadata else ''
                        
                        if grant_id and isinstance(keywords_result, dict):
                            extraction_data.append({
                                'grant_id': grant_id,
                                'keywords': keywords_result.get('keywords', []),
                                'methodology_keywords': keywords_result.get('methodology_keywords', []),
                                'application_keywords': keywords_result.get('application_keywords', []),
                                'technology_keywords': keywords_result.get('technology_keywords', []),
                                'total_keywords': len(keywords_result.get('keywords', [])) + 
                                               len(keywords_result.get('methodology_keywords', [])) + 
                                               len(keywords_result.get('application_keywords', [])) + 
                                               len(keywords_result.get('technology_keywords', []))
                            })
                    except Exception:
                        continue
            
            self.keyword_extractions = pd.DataFrame(extraction_data)
            return True
            
        except Exception as e:
            st.error(f"Error loading keyword extractions: {e}")
            return False
    
    def load_keyword_clusters(self, pattern: str = "*cluster*keywords*.eval") -> bool:
        """Load keyword clustering results from evaluation logs"""
        eval_files = list(self.logs_dir.glob(pattern))
        
        if not eval_files:
            st.warning("No keyword clustering evaluation files found")
            return False
        
        # Use the most recent file
        eval_file = max(eval_files, key=lambda x: x.stat().st_mtime)
        
        try:
            log = read_eval_log(str(eval_file))
            
            for sample in log.samples:
                if sample.output and sample.output.completion:
                    try:
                        clustering_result = json.loads(sample.output.completion)
                        
                        if 'topics' in clustering_result:
                            for topic_data in clustering_result['topics']:
                                topic_name = topic_data.get('topic_name', '')
                                self.keyword_clusters[topic_name] = {
                                    'description': topic_data.get('description', ''),
                                    'keywords': topic_data.get('keywords', []),
                                    'primary_keywords': topic_data.get('primary_keywords', []),
                                    'domain': topic_data.get('domain', None)
                                }
                        
                        break  # Take the first valid result
                        
                    except Exception:
                        continue
            
            return True
            
        except Exception as e:
            st.error(f"Error loading keyword clusters: {e}")
            return False
    
    def load_topic_assignments(self, assignment_file: str = "grant_topic_assignments.json") -> bool:
        """Load grant topic assignment results"""
        assignment_path = self.base_data_dir / assignment_file
        
        if not assignment_path.exists():
            st.warning(f"Topic assignment file not found: {assignment_path}")
            return False
        
        try:
            with open(assignment_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract assignments data
            assignments = data.get('assignments', [])
            self.assignment_summary = data.get('summary', {})
            
            # Convert to DataFrame for easier analysis
            assignment_rows = []
            for assignment in assignments:
                for topic in assignment['assigned_topics']:
                    assignment_rows.append({
                        'grant_id': assignment['grant_id'],
                        'grant_title': assignment['grant_title'],
                        'grant_summary': assignment['grant_summary'][:200] + '...' if len(assignment['grant_summary']) > 200 else assignment['grant_summary'],
                        'funding_amount': assignment.get('funding_amount', 0) or 0,
                        'topic': topic,
                        'topic_score': assignment['topic_scores'].get(topic, 0),
                        'num_keywords': len(assignment.get('extracted_keywords', [])),
                        'matched_keywords': ', '.join(assignment['matched_keywords'].get(topic, [])),
                        'num_matched_keywords': len(assignment['matched_keywords'].get(topic, []))
                    })
            
            self.topic_assignments = pd.DataFrame(assignment_rows)
            return True
            
        except Exception as e:
            st.error(f"Error loading topic assignments: {e}")
            return False
    
    def create_grants_overview_tab(self):
        """Create the grants data overview tab"""
        st.header("📊 Grants Data Overview")
        
        if self.grants_data is None:
            st.error("No grants data loaded")
            return
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Grants", f"{len(self.grants_data):,}")
        
        with col2:
            grants_with_summary = self.grants_data['has_summary'].sum()
            st.metric("Grants with Summary", f"{grants_with_summary:,}")
        
        with col3:
            total_funding = self.grants_data['funding_amount'].sum()
            st.metric("Total Funding", f"${total_funding/1e6:.1f}M")
        
        with col4:
            avg_funding = self.grants_data['funding_amount'].mean()
            st.metric("Average Funding", f"${avg_funding/1e3:.0f}K")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Funding Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            funding_data = self.grants_data['funding_amount'].dropna()
            if len(funding_data) > 0:
                ax.hist(funding_data / 1e3, bins=50, alpha=0.7, edgecolor='black')
                ax.set_xlabel('Funding Amount (Thousands AUD)')
                ax.set_ylabel('Number of Grants')
                ax.set_title('Distribution of Grant Funding')
                
            st.pyplot(fig)
        
        with col2:
            st.subheader("Grants by Funder")
            funder_counts = self.grants_data['funder'].value_counts().head(10)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(range(len(funder_counts)), funder_counts.values)
            ax.set_yticks(range(len(funder_counts)))
            ax.set_yticklabels([label[:30] + '...' if len(label) > 30 else label for label in funder_counts.index])
            ax.set_xlabel('Number of Grants')
            ax.set_title('Top 10 Funders by Grant Count')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                       f'{int(width)}', ha='left', va='center')
            
            st.pyplot(fig)
        
        # Summary statistics table
        st.subheader("Summary Statistics")
        summary_stats = self.grants_data[['funding_amount', 'summary_length']].describe()
        st.dataframe(summary_stats, use_container_width=True)
        
        # Sample grants
        st.subheader("Sample Grants")
        sample_grants = self.grants_data.sample(min(5, len(self.grants_data)))
        for _, grant in sample_grants.iterrows():
            with st.expander(f"{grant['title'][:100]}..."):
                st.write(f"**ID:** {grant['id']}")
                st.write(f"**Funder:** {grant.get('funder', 'N/A')}")
                st.write(f"**Funding:** ${grant.get('funding_amount', 0):,.0f}")
                st.write(f"**Summary:** {grant.get('grant_summary', 'No summary')[:300]}...")
    
    def create_keyword_extraction_tab(self):
        """Create the keyword extraction results tab"""
        st.header("🔑 Keyword Extraction Results")
        
        if self.keyword_extractions is None:
            st.error("No keyword extraction data loaded")
            return
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Grants Processed", len(self.keyword_extractions))
        
        with col2:
            avg_keywords = self.keyword_extractions['total_keywords'].mean()
            st.metric("Avg Keywords/Grant", f"{avg_keywords:.1f}")
        
        with col3:
            total_keywords = self.keyword_extractions['total_keywords'].sum()
            st.metric("Total Keywords", f"{total_keywords:,}")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Keywords per Grant Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            keyword_counts = self.keyword_extractions['total_keywords']
            ax.hist(keyword_counts, bins=30, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Number of Keywords')
            ax.set_ylabel('Number of Grants')
            ax.set_title('Distribution of Keywords per Grant')
            
            st.pyplot(fig)
        
        with col2:
            st.subheader("Keyword Categories Overview")
            
            # Calculate total keywords by category
            category_totals = {}
            for category in ['keywords', 'methodology_keywords', 'application_keywords', 'technology_keywords']:
                category_totals[category.replace('_keywords', '').replace('_', ' ').title()] = \
                    sum(len(kw_list) for kw_list in self.keyword_extractions[category])
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(category_totals.keys(), category_totals.values())
            ax.set_xlabel('Keyword Category')
            ax.set_ylabel('Total Keywords')
            ax.set_title('Keywords by Category')
            plt.xticks(rotation=45)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                       f'{int(height)}', ha='center', va='bottom')
            
            st.pyplot(fig)
        
        # Keyword categories analysis
        st.subheader("Keyword Categories Analysis")
        
        # Calculate average keywords per category
        category_stats = {}
        for category in ['keywords', 'methodology_keywords', 'application_keywords', 'technology_keywords']:
            lengths = self.keyword_extractions[category].apply(len)
            category_stats[category.replace('_', ' ').title()] = {
                'Average': lengths.mean(),
                'Total': lengths.sum(),
                'Max': lengths.max()
            }
        
        category_df = pd.DataFrame(category_stats).T
        st.dataframe(category_df, use_container_width=True)
        
        # Most common technology keywords
        st.subheader("Most Common Technology Keywords")
        
        # Flatten technology keywords and count occurrences
        tech_keywords = []
        for kw_list in self.keyword_extractions['technology_keywords']:
            tech_keywords.extend(kw_list)
        
        if tech_keywords:
            tech_counts = pd.Series(tech_keywords).value_counts().head(15)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.barh(range(len(tech_counts)), tech_counts.values)
            ax.set_yticks(range(len(tech_counts)))
            ax.set_yticklabels([label[:40] + '...' if len(label) > 40 else label for label in tech_counts.index])
            ax.set_xlabel('Frequency')
            ax.set_title('Top 15 Technology Keywords')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                       f'{int(width)}', ha='left', va='center')
            
            st.pyplot(fig)
        else:
            st.write("No technology keywords found.")
        
        # Sample extractions
        st.subheader("Sample Keyword Extractions")
        sample_extractions = self.keyword_extractions.sample(min(3, len(self.keyword_extractions)))
        
        for _, extraction in sample_extractions.iterrows():
            with st.expander(f"Grant {extraction['grant_id']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Total Keywords:** {extraction['total_keywords']}")
                    st.write(f"**General Keywords:** {', '.join(extraction['keywords'][:10])}")
                    if len(extraction['keywords']) > 10:
                        st.write(f"... and {len(extraction['keywords']) - 10} more")
                
                with col2:
                    st.write(f"**Methodology:** {', '.join(extraction['methodology_keywords'][:5])}")
                    if len(extraction['methodology_keywords']) > 5:
                        st.write(f"... and {len(extraction['methodology_keywords']) - 5} more")
                    
                    st.write(f"**Applications:** {', '.join(extraction['application_keywords'][:5])}")
                    if len(extraction['application_keywords']) > 5:
                        st.write(f"... and {len(extraction['application_keywords']) - 5} more")
                    
                    st.write(f"**Technologies:** {', '.join(extraction['technology_keywords'][:5])}")
                    if len(extraction['technology_keywords']) > 5:
                        st.write(f"... and {len(extraction['technology_keywords']) - 5} more")
    
    def create_keyword_clustering_tab(self):
        """Create the keyword clustering analysis tab"""
        st.header("🔗 Keyword Clustering Analysis")
        
        if not self.keyword_clusters:
            st.error("No keyword clustering data loaded")
            return
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Topic Clusters", len(self.keyword_clusters))
        
        with col2:
            avg_keywords = np.mean([len(cluster['keywords']) for cluster in self.keyword_clusters.values()])
            st.metric("Avg Keywords/Topic", f"{avg_keywords:.1f}")
        
        with col3:
            total_keywords = sum(len(cluster['keywords']) for cluster in self.keyword_clusters.values())
            st.metric("Total Keywords", total_keywords)
        
        # Cluster size distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cluster Size Distribution")
            cluster_sizes = [len(cluster['keywords']) for cluster in self.keyword_clusters.values()]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(cluster_sizes, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Number of Keywords')
            ax.set_ylabel('Number of Clusters')
            ax.set_title('Distribution of Keywords per Cluster')
            
            st.pyplot(fig)
        
        with col2:
            st.subheader("Confidence Distribution")
            confidences = [cluster['confidence'] for cluster in self.keyword_clusters.values()]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(confidences, bins=20, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Confidence Score')
            ax.set_ylabel('Number of Clusters')
            ax.set_title('Distribution of Cluster Confidence')
            
            st.pyplot(fig)
        
        # Top clusters by size
        st.subheader("Largest Topic Clusters")
        cluster_info = []
        for name, data in self.keyword_clusters.items():
            cluster_info.append({
                'Topic': name,
                'Keywords Count': len(data['keywords']),
                'Primary Keywords': len(data['primary_keywords']),
                'Confidence': data['confidence'],
                'Domain': data.get('domain', 'N/A')
            })
        
        cluster_df = pd.DataFrame(cluster_info)
        cluster_df = cluster_df.sort_values('Keywords Count', ascending=False)
        
        st.dataframe(cluster_df.head(15), use_container_width=True)
        
        # Detailed cluster view
        st.subheader("Detailed Cluster Information")
        
        # Select cluster to explore
        selected_cluster = st.selectbox(
            "Select a cluster to explore:",
            options=sorted(self.keyword_clusters.keys()),
            key="cluster_selection"
        )
        
        if selected_cluster:
            cluster_data = self.keyword_clusters[selected_cluster]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Description:** {cluster_data['description']}")
                st.write(f"**Confidence:** {cluster_data['confidence']:.3f}")
                st.write(f"**Domain:** {cluster_data.get('domain', 'N/A')}")
                st.write(f"**Total Keywords:** {len(cluster_data['keywords'])}")
                st.write(f"**Primary Keywords:** {len(cluster_data['primary_keywords'])}")
            
            with col2:
                st.write("**Primary Keywords:**")
                for keyword in cluster_data['primary_keywords']:
                    st.write(f"• {keyword}")
            
            st.write("**All Keywords:**")
            keywords_text = ', '.join(cluster_data['keywords'])
            st.text_area("Keywords", keywords_text, height=100, disabled=True)
    
    def create_topic_assignment_tab(self):
        """Create the topic assignment results tab"""
        st.header("🎯 Topic Assignment Results")
        
        if self.topic_assignments is None:
            st.error("No topic assignment data loaded")
            return
        
        # Basic statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Assigned Grants", len(self.topic_assignments['grant_id'].unique()))
        
        with col2:
            st.metric("Research Topics", len(self.topic_assignments['topic'].unique()))
        
        with col3:
            total_funding = self.topic_assignments['funding_amount'].sum()
            st.metric("Total Funding", f"${total_funding/1e6:.1f}M")
        
        with col4:
            avg_score = self.topic_assignments['topic_score'].mean()
            st.metric("Avg Topic Score", f"{avg_score:.3f}")
        
        # Summary from assignment results
        if self.assignment_summary:
            st.subheader("Assignment Summary")
            summary_col1, summary_col2 = st.columns(2)
            
            with summary_col1:
                st.metric("Total Grants Processed", self.assignment_summary.get('total_grants', 0))
                st.metric("Successfully Assigned", self.assignment_summary.get('assigned_grants', 0))
            
            with summary_col2:
                st.metric("Unassigned Grants", self.assignment_summary.get('unassigned_grants', 0))
                assignment_rate = (self.assignment_summary.get('assigned_grants', 0) / 
                                 max(self.assignment_summary.get('total_grants', 1), 1)) * 100
                st.metric("Assignment Rate", f"{assignment_rate:.1f}%")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top Topics by Grant Count")
            topic_counts = self.topic_assignments['topic'].value_counts().head(15)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.barh(range(len(topic_counts)), topic_counts.values)
            ax.set_yticks(range(len(topic_counts)))
            ax.set_yticklabels([label[:40] + '...' if len(label) > 40 else label for label in topic_counts.index])
            ax.set_xlabel('Number of Grants')
            ax.set_title('Top 15 Topics by Grant Count')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                       f'{int(width)}', ha='left', va='center')
            
            st.pyplot(fig)
        
        with col2:
            st.subheader("Top Topics by Funding")
            funding_by_topic = self.topic_assignments.groupby('topic')['funding_amount'].sum().sort_values(ascending=False).head(15)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            bars = ax.barh(range(len(funding_by_topic)), funding_by_topic.values / 1e6)
            ax.set_yticks(range(len(funding_by_topic)))
            ax.set_yticklabels([label[:40] + '...' if len(label) > 40 else label for label in funding_by_topic.index])
            ax.set_xlabel('Total Funding (Millions AUD)')
            ax.set_title('Top 15 Topics by Total Funding')
            
            # Add value labels
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                       f'${width:.1f}M', ha='left', va='center')
            
            st.pyplot(fig)
        
        # Topic score distribution
        st.subheader("Topic Score Distribution")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(self.topic_assignments['topic_score'], bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Topic Score')
        ax.set_ylabel('Number of Assignments')
        ax.set_title('Distribution of Topic Assignment Scores')
        st.pyplot(fig)
        
        # Detailed topic analysis
        st.subheader("Detailed Topic Analysis")
        
        # Select topic for detailed view
        available_topics = sorted(self.topic_assignments['topic'].unique())
        selected_topic = st.selectbox(
            "Select a topic for detailed analysis:",
            options=available_topics,
            key="topic_detail_selection"
        )
        
        if selected_topic:
            topic_data = self.topic_assignments[self.topic_assignments['topic'] == selected_topic]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Grants in Topic", len(topic_data))
            
            with col2:
                topic_funding = topic_data['funding_amount'].sum()
                st.metric("Total Funding", f"${topic_funding/1e6:.1f}M")
            
            with col3:
                avg_score = topic_data['topic_score'].mean()
                st.metric("Avg Assignment Score", f"{avg_score:.3f}")
            
            # Top grants in this topic
            st.write("**Top Grants by Assignment Score:**")
            top_grants = topic_data.nlargest(5, 'topic_score')
            
            for _, grant in top_grants.iterrows():
                with st.expander(f"Score: {grant['topic_score']:.3f} - {grant['grant_title'][:80]}..."):
                    st.write(f"**Grant ID:** {grant['grant_id']}")
                    st.write(f"**Funding:** ${grant['funding_amount']:,.0f}")
                    st.write(f"**Matched Keywords:** {grant['matched_keywords']}")
                    st.write(f"**Summary:** {grant['grant_summary']}")
        
        # Data table with filters
        st.subheader("Assignment Data Table")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            topic_filter = st.multiselect(
                "Filter by topics:",
                options=sorted(self.topic_assignments['topic'].unique()),
                key="assignment_topic_filter"
            )
        
        with col2:
            min_score = st.slider(
                "Minimum topic score:",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                key="assignment_score_filter"
            )
        
        with col3:
            min_funding = st.number_input(
                "Minimum funding (K):",
                min_value=0,
                value=0,
                step=50,
                key="assignment_funding_filter"
            ) * 1000
        
        # Apply filters
        filtered_data = self.topic_assignments.copy()
        
        if topic_filter:
            filtered_data = filtered_data[filtered_data['topic'].isin(topic_filter)]
        
        filtered_data = filtered_data[
            (filtered_data['topic_score'] >= min_score) & 
            (filtered_data['funding_amount'] >= min_funding)
        ]
        
        # Display filtered data
        st.dataframe(
            filtered_data.sort_values('topic_score', ascending=False),
            use_container_width=True,
            height=400
        )
        
        # Summary of filtered data
        if len(filtered_data) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Filtered Assignments", len(filtered_data))
            with col2:
                st.metric("Unique Grants", filtered_data['grant_id'].nunique())
            with col3:
                st.metric("Total Funding", f"${filtered_data['funding_amount'].sum()/1e6:.1f}M")


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Research Grant Analysis Workflow Dashboard",
        page_icon="🔬",
        layout="wide"
    )
    
    st.title("🔬 Research Grant Analysis Workflow Dashboard")
    st.markdown("**Comprehensive visualization of the keyword-based research grant analysis pipeline**")
    st.markdown("---")
    
    # Initialize analyzer
    analyzer = WorkflowAnalyzer()
    
    # Sidebar with data loading status
    with st.sidebar:
        st.header("📁 Data Loading Status")
        
        data_status = {}
        
        # Load all data sources
        with st.spinner("Loading workflow data..."):
            data_status['grants'] = analyzer.load_grants_data()
            data_status['extractions'] = analyzer.load_keyword_extractions()
            data_status['clusters'] = analyzer.load_keyword_clusters()
            data_status['assignments'] = analyzer.load_topic_assignments()
        
        # Display status
        status_icons = {True: "✅", False: "❌"}
        st.write(f"{status_icons[data_status['grants']]} Grants Data")
        st.write(f"{status_icons[data_status['extractions']]} Keyword Extractions")
        st.write(f"{status_icons[data_status['clusters']]} Keyword Clustering")
        st.write(f"{status_icons[data_status['assignments']]} Topic Assignments")
        
        st.markdown("---")
        
        # Workflow progress
        completed_steps = sum(data_status.values())
        st.metric("Workflow Progress", f"{completed_steps}/4 Steps")
        
        if completed_steps < 4:
            st.warning("⚠️ Some workflow steps are incomplete")
            st.markdown("""
            **To complete the workflow:**
            1. Extract keywords: `inspect eval keywords_extraction.py`
            2. Cluster keywords: `inspect eval keywords_clustering.py`
            3. Assign topics: `python topic_classification.py`
            """)
    
    # Create tabs for each workflow step
    tab_names = [
        "📊 Grants Data", 
        "🔑 Keyword Extraction", 
        "🔗 Keyword Clustering", 
        "🎯 Topic Assignment"
    ]
    tabs = st.tabs(tab_names)
    
    # Grants Data Tab
    with tabs[0]:
        if data_status['grants']:
            analyzer.create_grants_overview_tab()
        else:
            st.error("Please ensure the grants data file exists in the data directory")
    
    # Keyword Extraction Tab
    with tabs[1]:
        if data_status['extractions']:
            analyzer.create_keyword_extraction_tab()
        else:
            st.error("No keyword extraction results found. Please run the keyword extraction task first.")
    
    # Keyword Clustering Tab
    with tabs[2]:
        if data_status['clusters']:
            analyzer.create_keyword_clustering_tab()
        else:
            st.error("No keyword clustering results found. Please run the keyword clustering task first.")
    
    # Topic Assignment Tab
    with tabs[3]:
        if data_status['assignments']:
            analyzer.create_topic_assignment_tab()
        else:
            st.error("No topic assignment results found. Please run the topic classification script first.")


if __name__ == "__main__":
    main()
