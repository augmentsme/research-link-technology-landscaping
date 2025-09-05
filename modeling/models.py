
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Optional





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



class FORDefinition(BaseModel):
    """Definition structure for FOR codes with description and exclusions."""
    model_config = {"extra": "forbid"}
    description: str = Field(description="The main description of the FOR code")
    exclusions: Optional[List[str]] = Field(default=None, description="List of exclusions for this FOR code")


class FORField(BaseModel):
    """Individual FOR field (6-digit code)."""
    model_config = {"extra": "forbid"}
    code: str = Field(description="6-digit FOR field code")
    name: str = Field(description="Name of the FOR field")
    definition: Optional[FORDefinition] = Field(default=None, description="Field definition with description and exclusions")
    exclusions: Optional[List[str]] = Field(default=None, description="Field-specific exclusions")


class FORGroup(BaseModel):
    """FOR group (4-digit code) containing multiple fields."""
    model_config = {"extra": "forbid"}
    code: str = Field(description="4-digit FOR group code")
    name: str = Field(description="Name of the FOR group")
    definition: Optional[FORDefinition] = Field(default=None, description="Group definition with description and exclusions")
    exclusions: Optional[List[str]] = Field(default=None, description="Group-specific exclusions")
    fields: Dict[str, FORField] = Field(default_factory=dict, description="Dictionary of fields in this group, keyed by field code")


class FORDivision(BaseModel):
    """FOR division (2-digit code) containing multiple groups."""
    model_config = {"extra": "forbid"}
    code: str = Field(description="2-digit FOR division code")
    name: str = Field(description="Name of the FOR division")
    definition: Optional[FORDefinition] = Field(default=None, description="Division definition with description and exclusions")
    exclusions: Optional[List[str]] = Field(default=None, description="Division-specific exclusions")
    groups: Dict[str, FORGroup] = Field(default_factory=dict, description="Dictionary of groups in this division, keyed by group code")




