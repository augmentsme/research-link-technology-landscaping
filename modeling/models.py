
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Optional



class KeywordType(str, Enum):
    GENERAL = "General"
    METHODOLOGY = "Methodology"
    APPLICATION = "Application"
    TECHNOLOGY = "Technology"

class FieldOfResearch(str, Enum):
    """Enumeration of Fields of Research (FOR) division names."""
    AGRICULTURAL_VETERINARY_FOOD_SCIENCES = "AGRICULTURAL_VETERINARY_FOOD_SCIENCES"
    BIOLOGICAL_SCIENCES = "BIOLOGICAL_SCIENCES"
    BIOMEDICAL_CLINICAL_SCIENCES = "BIOMEDICAL_CLINICAL_SCIENCES"
    BUILT_ENVIRONMENT_DESIGN = "BUILT_ENVIRONMENT_DESIGN"
    CHEMICAL_SCIENCES = "CHEMICAL_SCIENCES"
    COMMERCE_MANAGEMENT_TOURISM_SERVICES = "COMMERCE_MANAGEMENT_TOURISM_SERVICES"
    CREATIVE_ARTS_WRITING = "CREATIVE_ARTS_WRITING"
    EARTH_SCIENCES = "EARTH_SCIENCES"
    ECONOMICS = "ECONOMICS"
    EDUCATION = "EDUCATION"
    ENGINEERING = "ENGINEERING"
    ENVIRONMENTAL_SCIENCES = "ENVIRONMENTAL_SCIENCES"
    HEALTH_SCIENCES = "HEALTH_SCIENCES"
    HISTORY_HERITAGE_ARCHAEOLOGY = "HISTORY_HERITAGE_ARCHAEOLOGY"
    HUMAN_SOCIETY = "HUMAN_SOCIETY"
    INDIGENOUS_STUDIES = "INDIGENOUS_STUDIES"
    INFORMATION_COMPUTING_SCIENCES = "INFORMATION_COMPUTING_SCIENCES"
    LANGUAGE_COMMUNICATION_CULTURE = "LANGUAGE_COMMUNICATION_CULTURE"
    LAW_LEGAL_STUDIES = "LAW_LEGAL_STUDIES"
    MATHEMATICAL_SCIENCES = "MATHEMATICAL_SCIENCES"
    PHILOSOPHY_RELIGIOUS_STUDIES = "PHILOSOPHY_RELIGIOUS_STUDIES"
    PHYSICAL_SCIENCES = "PHYSICAL_SCIENCES"
    PSYCHOLOGY = "PSYCHOLOGY"
    
class Keyword(BaseModel):
    """Individual keyword with context."""
    model_config = {"extra": "forbid"}
    name: str = Field(description="The actual keyword or phrase")
    type: KeywordType = Field(description="Type of keyword: general, methodology, application, or technology")
    description: str = Field(description="Short description explaining the context and relevance of this keyword within the research")
    # grants: List[str] = Field(description="List of grant IDs where this keyword appears")

class KeywordsList(BaseModel):
    """Pydantic model for structured keywords extraction output."""
    model_config = {"extra": "forbid"}
    keywords: List[Keyword] = Field(description="List of all extracted keywords with their types and descriptions")

class Category(BaseModel):
    """A flexible research category linked to FOR codes."""
    model_config = {"extra": "forbid"}
    name: str = Field(description="Name of the category")
    description: str = Field(description="A few sentences describing what this category is about, including its scope, focus areas, and the types of research or technologies it encompasses")
    keywords: List[str] = Field(description="List of keywords associated with this category")
    field_of_research: FieldOfResearch = Field(description="The field of research division this category falls under")

class CategoryList(BaseModel):
    """A list of research categories."""
    model_config = {"extra": "forbid"}
    categories: List[Category] = Field(description="List of research categories")
