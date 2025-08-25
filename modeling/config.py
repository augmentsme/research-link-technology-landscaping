"""
Application Configuration Module

This module provides a centralized configuration system that reads from YAML and .env files.
It replaces the legacy config.py approach with a cleaner, more maintainable structure.
"""

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass
class Config:
    """Configuration class that encapsulates all application settings."""
    # Base directories
    root_dir: Path
    data_dir: Path
    results_dir: Path
    figures_dir: Path
    prompts_dir: Path
    
    # Data paths
    grants_file: Path
    for_codes_cleaned_path: Path
    
    # Keywords configuration
    extracted_keywords_dir: Path
    extracted_keywords_path: Path
    keywords_path: Path
    chromadb_path: Path
    batch_size: int
    keywords_type: Optional[str]
    similarity_threshold: float
    min_cluster_size: int
    
    # Grant paths (legacy)
    grants_use_embeddings: bool
    
    # Clustering paths
    clusters_proposal_path: Path
    clusters_final_path: Path
    review_file: Path
    
    # Categories paths
    category_proposal_path: Path
    category_path: Path
    coarsened_category_path: Path
    refined_category_path: Path
    comprehensive_taxonomy_path: Path
    
    # Classification paths
    classification_path: Path
    
    # vLLM configuration
    vllm_base_url: str
    vllm_generation_model: str
    vllm_embedding_model: str
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None, env_path: Optional[Path] = None) -> 'Config':
        """Load configuration from YAML and .env files."""
        # Set default paths
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        # Load YAML configuration
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        
        root_dir = Path(__file__).parent

        # Derive other directories from ROOT_DIR
        data_dir = (root_dir / "data").resolve()
        results_dir = (root_dir / "results").resolve()
        figures_dir = (root_dir / "figures").resolve()
        prompts_dir = (root_dir / yaml_config["prompts"]["dir"]).resolve()
        
        return cls(
            # Base directories
            root_dir=root_dir,
            data_dir=data_dir,
            results_dir=results_dir,
            figures_dir=figures_dir,
            prompts_dir=prompts_dir,
            

            chromadb_path=results_dir / yaml_config["chromadb_path"],

            # Data paths
            grants_file=data_dir / yaml_config["data"]["grants_file"],
            for_codes_cleaned_path=data_dir / yaml_config["data"]["for_codes_cleaned_file"],
            
            # Keywords configuration
            extracted_keywords_dir=results_dir / yaml_config["keywords"]["extracted_dir"],
            extracted_keywords_path=results_dir / yaml_config["keywords"]["extracted_file"],
            keywords_path=results_dir / yaml_config["keywords"]["keywords_file"],
            batch_size=yaml_config["keywords"]["batch_size"],
            keywords_type=yaml_config["keywords"]["keywords_type"],
            similarity_threshold=yaml_config["keywords"]["similarity_threshold"],
            min_cluster_size=yaml_config["keywords"]["min_cluster_size"],
            
            # Grant paths (legacy)
            grants_use_embeddings=yaml_config["grants"]["use_embeddings"],
            
            # Clustering paths
            clusters_proposal_path=results_dir / yaml_config["clustering"]["proposal_file"],
            clusters_final_path=results_dir / yaml_config["clustering"]["final_file"],
            review_file=results_dir / yaml_config["clustering"]["review_file"],
            
            # Categories paths
            category_proposal_path=results_dir / yaml_config["categories"]["proposal_file"],
            category_path=results_dir / yaml_config["categories"]["categories_file"],
            coarsened_category_path=results_dir / yaml_config["categories"]["coarsened_file"],
            refined_category_path=results_dir / yaml_config["categories"]["refined_file"],
            comprehensive_taxonomy_path=results_dir / yaml_config["categories"]["comprehensive_taxonomy_file"],
            
            # Classification paths
            classification_path=results_dir / yaml_config["classification"]["classification_file"],
            
            # vLLM configuration
            vllm_base_url=yaml_config["vllm"]["base_url"],
            vllm_generation_model=yaml_config["vllm"]["generation_model"],
            vllm_embedding_model=yaml_config["vllm"]["embedding_model"],
        )
    
    def get_vllm_config(self) -> Dict:
        """Get vLLM configuration as dictionary."""
        return {
            "base_url": self.vllm_base_url,
            "generation_model": self.vllm_generation_model,
            "embedding_model": self.vllm_embedding_model,
        }
    
    def get_keywords_config(self) -> Dict:
        """Get keywords configuration as dictionary."""
        return {
            "batch_size": self.batch_size,
            "keywords_type": self.keywords_type,
            "similarity_threshold": self.similarity_threshold,
            "min_cluster_size": self.min_cluster_size,
        }
    
    def get_grants_config(self) -> Dict:
        """Get grants configuration as dictionary."""
        return {
            "use_embeddings": self.grants_use_embeddings,
        }


# Global config instance
CONFIG = Config.load()
