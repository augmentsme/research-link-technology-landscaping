
from enum import Enum
from pydantic import BaseModel, Field
from typing import List

class KeywordType(str, Enum):
    """Enumeration of valid keyword types."""
    GENERAL = "General"
    METHODOLOGY = "Methodology"
    APPLICATION = "Application"
    TECHNOLOGY = "Technology"
    PROBLEM = "Problem"

class FORCode(str, Enum):
    """Enumeration of 2-digit Fields of Research (FOR) codes."""
    AGRICULTURAL_VETERINARY_FOOD_SCIENCES = "30"
    BIOLOGICAL_SCIENCES = "31"
    BIOMEDICAL_CLINICAL_SCIENCES = "32"
    BUILT_ENVIRONMENT_DESIGN = "33"
    CHEMICAL_SCIENCES = "34"
    COMMERCE_MANAGEMENT_TOURISM_SERVICES = "35"
    CREATIVE_ARTS_WRITING = "36"
    EARTH_SCIENCES = "37"
    ECONOMICS = "38"
    EDUCATION = "39"
    ENGINEERING = "40"
    ENVIRONMENTAL_SCIENCES = "41"
    HEALTH_SCIENCES = "42"
    HISTORY_HERITAGE_ARCHAEOLOGY = "43"
    HUMAN_SOCIETY = "44"
    INDIGENOUS_STUDIES = "45"
    INFORMATION_COMPUTING_SCIENCES = "46"
    LANGUAGE_COMMUNICATION_CULTURE = "47"
    LAW_LEGAL_STUDIES = "48"
    MATHEMATICAL_SCIENCES = "49"
    PHILOSOPHY_RELIGIOUS_STUDIES = "50"
    PHYSICAL_SCIENCES = "51"
    PSYCHOLOGY = "52"
    
    @classmethod
    def get_name(cls, code_value: str) -> str:
        """Get the human-readable name for a FOR code."""
        name_mapping = {
            "30": "AGRICULTURAL, VETERINARY AND FOOD SCIENCES",
            "31": "BIOLOGICAL SCIENCES",
            "32": "BIOMEDICAL AND CLINICAL SCIENCES",
            "33": "BUILT ENVIRONMENT AND DESIGN",
            "34": "CHEMICAL SCIENCES",
            "35": "COMMERCE, MANAGEMENT, TOURISM AND SERVICES",
            "36": "CREATIVE ARTS AND WRITING",
            "37": "EARTH SCIENCES",
            "38": "ECONOMICS",
            "39": "EDUCATION",
            "40": "ENGINEERING",
            "41": "ENVIRONMENTAL SCIENCES",
            "42": "HEALTH SCIENCES",
            "43": "HISTORY, HERITAGE AND ARCHAEOLOGY",
            "44": "HUMAN SOCIETY",
            "45": "INDIGENOUS STUDIES",
            "46": "INFORMATION AND COMPUTING SCIENCES",
            "47": "LANGUAGE, COMMUNICATION AND CULTURE",
            "48": "LAW AND LEGAL STUDIES",
            "49": "MATHEMATICAL SCIENCES",
            "50": "PHILOSOPHY AND RELIGIOUS STUDIES",
            "51": "PHYSICAL SCIENCES",
            "52": "PSYCHOLOGY"
        }
        return name_mapping.get(code_value, f"Unknown FOR Code: {code_value}")



class Keyword(BaseModel):
    """Individual keyword with context."""
    model_config = {"extra": "forbid"}
    term: str = Field(description="The actual keyword or phrase")
    type: KeywordType = Field(description="Type of keyword: general, methodology, application, or technology")
    description: str = Field(description="Short description explaining the context and relevance of this keyword within the research")
    grants: List[str] = Field(description="List of grant IDs where this keyword appears")

class KeywordsExtractionOutput(BaseModel):
    """Pydantic model for structured keywords extraction output."""
    model_config = {"extra": "forbid"}
    keywords: List[Keyword] = Field(description="List of all extracted keywords with their types and descriptions")

class Category(BaseModel):
    """A flexible research category linked to FOR codes."""
    model_config = {"extra": "forbid"}
    name: str = Field(description="Name of the category")
    description: str = Field(description="A few sentences describing what this category is about, including its scope, focus areas, and the types of research or technologies it encompasses")
    keywords: List[str] = Field(description="List of keywords associated with this category")
    for_code: FORCode = Field(description="The 2-digit FOR code this category falls under")

class CategoryList(BaseModel):
    """A list of research categories."""
    model_config = {"extra": "forbid"}
    categories: List[Category] = Field(description="List of research categories")

