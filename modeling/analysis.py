%matplotlib inline
"""
Object-oriented analysis of topic classification results with visualizations
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
                
        except Exception:
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
        
    def get_metrics(self) -> Dict[str, float]:
        """Calculate key performance metrics"""
        if self.analysis_df is None:
            raise ValueError("Data not processed. Call process_responses() first.")
            
        total_grants = len(self.analysis_df)
        successful_classifications = self.analysis_df['predicted_id'].notna().sum()
        unique_predictions = self.analysis_df['predicted_id'].nunique()
        total_available = len(self.id_to_name) if self.id_to_name else 0
        
        return {
            'total_grants': int(total_grants),
            'successful_classifications': int(successful_classifications),
            'success_rate': float(successful_classifications / total_grants * 100),
            'unique_predictions': int(unique_predictions),
            'coverage_rate': float(unique_predictions / total_available * 100) if total_available > 0 else 0.0,
            'total_available_subfields': int(total_available)
        }
        
    def get_prediction_distribution(self) -> pd.Series:
        """Get distribution of predictions by subfield"""
        if self.analysis_df is None:
            raise ValueError("Data not processed. Call process_responses() first.")
            
        if 'predicted_name' in self.analysis_df.columns:
            return self.analysis_df['predicted_name'].value_counts()
        else:
            return self.analysis_df['predicted_id'].value_counts()
            
    def get_domain_analysis(self) -> Dict[str, pd.Series]:
        """Analyze predictions by OpenAlex domain"""
        if self.analysis_df is None:
            raise ValueError("Data not processed. Call process_responses() first.")
            
        domain_analysis = {}
        
        # Group by predicted domain
        if 'predicted_domain' in self.analysis_df.columns:
            domain_grants = self.analysis_df.dropna(subset=['predicted_domain'])
            
            # Get unique domains
            unique_domains = domain_grants['predicted_domain'].unique()
            
            for domain in unique_domains:
                domain_mask = domain_grants['predicted_domain'] == domain
                domain_data = domain_grants[domain_mask]
                
                if 'predicted_name' in domain_data.columns:
                    domain_analysis[domain] = domain_data['predicted_name'].value_counts()
                else:
                    domain_analysis[domain] = domain_data['predicted_id'].value_counts()
        
        return domain_analysis
        
    def get_field_analysis(self) -> Dict[str, pd.Series]:
        """Analyze predictions by OpenAlex field"""
        if self.analysis_df is None:
            raise ValueError("Data not processed. Call process_responses() first.")
            
        field_analysis = {}
        
        # Group by predicted field
        if 'predicted_field' in self.analysis_df.columns:
            field_grants = self.analysis_df.dropna(subset=['predicted_field'])
            
            # Get unique fields
            unique_fields = field_grants['predicted_field'].unique()
            
            for field in unique_fields:
                field_mask = field_grants['predicted_field'] == field
                field_data = field_grants[field_mask]
                
                if 'predicted_name' in field_data.columns:
                    field_analysis[field] = field_data['predicted_name'].value_counts()
                else:
                    field_analysis[field] = field_data['predicted_id'].value_counts()
        
        return field_analysis
        
    def create_visualizations(self, output_dir: str = "plots") -> None:
        """Create comprehensive visualizations of the analysis"""
        if self.analysis_df is None:
            raise ValueError("Data not processed. Call process_responses() first.")
            
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Top subfields pie chart
        self._plot_top_subfields_pie(output_dir)
        
        # 2. Top subfields bar chart
        self._plot_top_subfields_bar(output_dir)
        
        # 3. Domain distribution
        self._plot_domain_distribution(output_dir)
        
        # 4. Field distribution  
        self._plot_field_distribution(output_dir)
        
        # 5. Coverage analysis
        self._plot_coverage_analysis(output_dir)
        
        # 6. Prediction frequency distribution
        self._plot_prediction_frequency_distribution(output_dir)
        
    def _plot_top_subfields_pie(self, output_dir: str) -> None:
        """Create pie chart of top predicted subfields"""
        prediction_dist = self.get_prediction_distribution()
        top_10 = prediction_dist.head(10)
        others_count = prediction_dist.iloc[10:].sum()
        
        # Prepare data for pie chart
        if others_count > 0:
            pie_data = pd.concat([top_10, pd.Series([others_count], index=['Others'])])
        else:
            pie_data = top_10
            
        plt.figure(figsize=(12, 8))
        colors = sns.color_palette("husl", len(pie_data))
        
        wedges, texts, autotexts = plt.pie(pie_data.values, labels=pie_data.index, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        
        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            
        plt.title('Distribution of Research Funding by Subfield\n(Top 10 + Others)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/subfields_pie_chart.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_top_subfields_bar(self, output_dir: str) -> None:
        """Create horizontal bar chart of top subfields"""
        prediction_dist = self.get_prediction_distribution()
        top_15 = prediction_dist.head(15)
        
        plt.figure(figsize=(12, 10))
        colors = sns.color_palette("viridis", len(top_15))
        
        bars = plt.barh(range(len(top_15)), top_15.values, color=colors)
        plt.yticks(range(len(top_15)), top_15.index)
        plt.xlabel('Number of Grants', fontsize=12)
        plt.title('Top 15 Research Subfields by Number of Grants', fontsize=16, fontweight='bold')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_15.values)):
            plt.text(value + 0.1, i, f'{value}', va='center', fontweight='bold')
            
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/subfields_bar_chart.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_domain_distribution(self, output_dir: str) -> None:
        """Create domain distribution visualization using OpenAlex domains"""
        domain_analysis = self.get_domain_analysis()
        
        if not domain_analysis:
            return
        
        # Calculate total grants per domain
        domain_totals = {domain: predictions.sum() for domain, predictions in domain_analysis.items()}
        domain_totals_series = pd.Series(domain_totals).sort_values(ascending=False)
        
        plt.figure(figsize=(12, 8))
        colors = sns.color_palette("Set2", len(domain_totals_series))
        
        bars = plt.bar(domain_totals_series.index, domain_totals_series.values, color=colors)
        plt.xlabel('OpenAlex Domain', fontsize=12)
        plt.ylabel('Number of Grants', fontsize=12)
        plt.title('Research Funding Distribution by OpenAlex Domain', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, domain_totals_series.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value}', ha='center', va='bottom', fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(f"{output_dir}/domain_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_field_distribution(self, output_dir: str) -> None:
        """Create field distribution visualization using OpenAlex fields"""
        field_analysis = self.get_field_analysis()
        
        if not field_analysis:
            return
        
        # Calculate total grants per field
        field_totals = {field: predictions.sum() for field, predictions in field_analysis.items()}
        field_totals_series = pd.Series(field_totals).sort_values(ascending=False)
        
        plt.figure(figsize=(14, 8))
        colors = sns.color_palette("Set3", len(field_totals_series))
        
        bars = plt.bar(field_totals_series.index, field_totals_series.values, color=colors)
        plt.xlabel('OpenAlex Field', fontsize=12)
        plt.ylabel('Number of Grants', fontsize=12)
        plt.title('Research Funding Distribution by OpenAlex Field', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, field_totals_series.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{value}', ha='center', va='bottom', fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(f"{output_dir}/field_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_coverage_analysis(self, output_dir: str) -> None:
        """Create coverage analysis visualization"""
        metrics = self.get_metrics()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Success rate pie chart
        success_data = [metrics['success_rate'], 100 - metrics['success_rate']]
        success_labels = ['Successful', 'Failed']
        colors1 = ['#2ecc71', '#e74c3c']
        
        ax1.pie(success_data, labels=success_labels, autopct='%1.1f%%', colors=colors1, startangle=90)
        ax1.set_title('Classification Success Rate', fontsize=14, fontweight='bold')
        
        # Coverage analysis
        coverage_data = [metrics['coverage_rate'], 100 - metrics['coverage_rate']]
        coverage_labels = ['Used Subfields', 'Unused Subfields']
        colors2 = ['#3498db', '#95a5a6']
        
        ax2.pie(coverage_data, labels=coverage_labels, autopct='%1.1f%%', colors=colors2, startangle=90)
        ax2.set_title('Subfield Coverage\n(Used vs Available)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/coverage_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_prediction_frequency_distribution(self, output_dir: str) -> None:
        """Create histogram of prediction frequencies"""
        prediction_dist = self.get_prediction_distribution()
        
        plt.figure(figsize=(10, 6))
        
        # Create histogram of prediction counts
        plt.hist(prediction_dist.values, bins=range(1, max(prediction_dist.values) + 2), 
                alpha=0.7, color='skyblue', edgecolor='black')
        
        plt.xlabel('Number of Grants per Subfield', fontsize=12)
        plt.ylabel('Number of Subfields', fontsize=12)
        plt.title('Distribution of Grant Counts Across Subfields', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        mean_grants = prediction_dist.mean()
        median_grants = prediction_dist.median()
        plt.axvline(mean_grants, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_grants:.1f}')
        plt.axvline(median_grants, color='orange', linestyle='--', alpha=0.8, label=f'Median: {median_grants:.1f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/prediction_frequency_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_results(self, output_file: str = "classification_results_detailed.csv") -> None:
        """Save detailed results to CSV"""
        if self.analysis_df is None:
            raise ValueError("Data not processed. Call process_responses() first.")
            
        # Select relevant columns
        columns = ['sample_id', 'metadata_grant_title', 'metadata_grant_summary', 
                  'predicted_id', 'metadata_classification_type']
        
        if 'predicted_name' in self.analysis_df.columns:
            columns.append('predicted_name')
        if 'predicted_domain' in self.analysis_df.columns:
            columns.append('predicted_domain')
        if 'predicted_field' in self.analysis_df.columns:
            columns.append('predicted_field')
            
        results = self.analysis_df[columns].copy()
        results.to_csv(output_file, index=False)
        
    def generate_summary_report(self) -> Dict:
        """Generate comprehensive summary report"""
        if self.analysis_df is None:
            raise ValueError("Data not processed. Call process_responses() first.")
            
        metrics = self.get_metrics()
        prediction_dist = self.get_prediction_distribution()
        domain_analysis = self.get_domain_analysis()
        field_analysis = self.get_field_analysis()
        
        # Top predictions
        top_predictions = []
        for name, count in prediction_dist.head(10).items():
            percentage = count / metrics['total_grants'] * 100
            top_predictions.append({
                'subfield': name,
                'grants': int(count),
                'percentage': float(percentage)
            })
            
        # Domain summary
        domain_summary = {}
        for domain, predictions in domain_analysis.items():
            domain_summary[domain] = {
                'total_grants': int(predictions.sum()),
                'percentage': float(predictions.sum() / metrics['total_grants'] * 100),
                'top_subfield': predictions.index[0] if len(predictions) > 0 else None,
                'top_subfield_count': int(predictions.iloc[0]) if len(predictions) > 0 else 0
            }
            
        # Field summary
        field_summary = {}
        for field, predictions in field_analysis.items():
            field_summary[field] = {
                'total_grants': int(predictions.sum()),
                'percentage': float(predictions.sum() / metrics['total_grants'] * 100),
                'top_subfield': predictions.index[0] if len(predictions) > 0 else None,
                'top_subfield_count': int(predictions.iloc[0]) if len(predictions) > 0 else 0
            }
            
        return {
            'metrics': metrics,
            'top_predictions': top_predictions,
            'domain_summary': domain_summary,
            'field_summary': field_summary,
            'eval_info': {
                'eval_id': self.samples.iloc[0].eval_id,
                'classification_type': self.samples.iloc[0].metadata_classification_type,
                'template_type': self.samples.iloc[0].metadata_template_type
            }
        }



analyzer = TopicClassificationAnalyzer("logs/subfields-100.eval")

# Run analysis
analyzer.load_data()
analyzer.process_responses()

# Generate visualizations
analyzer.create_visualizations()

# Save results
# analyzer.save_results()

# Generate and save summary report
# summary = analyzer.generate_summary_report()



# if __name__ == "__main__":
#     main()

