
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

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


class FORCodesHierarchy(BaseModel):
    """Complete hierarchical structure of FOR codes: divisions -> groups -> fields."""
    model_config = {"extra": "forbid"}
    divisions: Dict[str, FORDivision] = Field(description="Dictionary of all FOR divisions, keyed by division code")
    
    def get_division(self, code: str) -> Optional[FORDivision]:
        """Get a division by its 2-digit code."""
        return self.divisions.get(code)
    
    def get_group(self, code: str) -> Optional[FORGroup]:
        """Get a group by its 4-digit code."""
        if len(code) < 4:
            return None
        div_code = code[:2]
        division = self.get_division(div_code)
        return division.groups.get(code) if division else None
    
    def get_field(self, code: str) -> Optional[FORField]:
        """Get a field by its 6-digit code."""
        if len(code) < 6:
            return None
        group_code = code[:4]
        group = self.get_group(group_code)
        return group.fields.get(code) if group else None
    
    def get_name_by_code(self, code: str) -> str:
        """Get the name for any FOR code (2, 4, or 6 digits)."""
        if len(code) == 2:
            division = self.get_division(code)
            return division.name if division else f"Unknown Division: {code}"
        elif len(code) == 4:
            group = self.get_group(code)
            return group.name if group else f"Unknown Group: {code}"
        elif len(code) == 6:
            field = self.get_field(code)
            return field.name if field else f"Unknown Field: {code}"
        else:
            return f"Invalid FOR Code: {code}"
    
    def get_all_codes_at_level(self, level: int) -> List[str]:
        """Get all codes at a specific level (2=divisions, 4=groups, 6=fields)."""
        if level == 2:
            return list(self.divisions.keys())
        elif level == 4:
            codes = []
            for division in self.divisions.values():
                codes.extend(division.groups.keys())
            return codes
        elif level == 6:
            codes = []
            for division in self.divisions.values():
                for group in division.groups.values():
                    codes.extend(group.fields.keys())
            return codes
        else:
            return []

    def get_all_groups(self) -> Dict[str, FORGroup]:
        """Get all groups as a flat dictionary keyed by group code."""
        groups = {}
        for division in self.divisions.values():
            groups.update(division.groups)
        return groups

    def get_all_fields(self) -> Dict[str, FORField]:
        """Get all fields as a flat dictionary keyed by field code."""
        fields = {}
        for division in self.divisions.values():
            for group in division.groups.values():
                fields.update(group.fields)
        return fields
    
    def get_hierarchy_path(self, code: str) -> Dict[str, str]:
        """Get the full hierarchy path for a code."""
        if len(code) == 6:
            field = self.get_field(code)
            group = self.get_group(code[:4])
            division = self.get_division(code[:2])
            return {
                "division": f"{division.code}: {division.name}" if division else "Unknown",
                "group": f"{group.code}: {group.name}" if group else "Unknown", 
                "field": f"{field.code}: {field.name}" if field else "Unknown"
            }
        elif len(code) == 4:
            group = self.get_group(code)
            division = self.get_division(code[:2])
            return {
                "division": f"{division.code}: {division.name}" if division else "Unknown",
                "group": f"{group.code}: {group.name}" if group else "Unknown"
            }
        elif len(code) == 2:
            division = self.get_division(code)
            return {
                "division": f"{division.code}: {division.name}" if division else "Unknown"
            }
        else:
            return {"error": f"Invalid FOR code: {code}"}

    @classmethod
    def from_json_file(cls, file_path: str) -> 'FORCodesHierarchy':
        """Load FOR codes hierarchy from JSON file."""
        import json
        from pathlib import Path
        
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"FOR codes file not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls.from_raw_data(data)
    
    @classmethod
    def from_raw_data(cls, raw_data: dict) -> 'FORCodesHierarchy':
        """Convert raw FOR codes data to structured Pydantic model."""
        divisions = {}
        
        for div_code, div_data in raw_data.items():
            # Process division definition
            div_definition = None
            if 'definition' in div_data and div_data['definition']:
                div_def_data = div_data['definition']
                if isinstance(div_def_data, dict):
                    div_definition = FORDefinition(
                        description=div_def_data.get('description', ''),
                        exclusions=div_def_data.get('exclusions', [])
                    )
            
            # Process groups
            groups = {}
            if 'groups' in div_data:
                for group_code, group_data in div_data['groups'].items():
                    # Process group definition
                    group_definition = None
                    if 'definition' in group_data and group_data['definition']:
                        group_def_data = group_data['definition']
                        if isinstance(group_def_data, dict):
                            group_definition = FORDefinition(
                                description=group_def_data.get('description', ''),
                                exclusions=group_def_data.get('exclusions', [])
                            )
                    
                    # Process fields
                    fields = {}
                    if 'fields' in group_data:
                        for field_code, field_data in group_data['fields'].items():
                            # Process field definition
                            field_definition = None
                            if 'definition' in field_data and field_data['definition']:
                                field_def_data = field_data['definition']
                                if isinstance(field_def_data, dict):
                                    field_definition = FORDefinition(
                                        description=field_def_data.get('description', ''),
                                        exclusions=field_def_data.get('exclusions', [])
                                    )
                            
                            fields[field_code] = FORField(
                                code=field_code,
                                name=field_data.get('name', ''),
                                definition=field_definition,
                                exclusions=field_data.get('exclusions', [])
                            )
                    
                    groups[group_code] = FORGroup(
                        code=group_code,
                        name=group_data.get('name', ''),
                        definition=group_definition,
                        exclusions=group_data.get('exclusions', []),
                        fields=fields
                    )
            
            divisions[div_code] = FORDivision(
                code=div_code,
                name=div_data.get('name', ''),
                definition=div_definition,
                exclusions=div_data.get('exclusions', []),
                groups=groups
            )
        
        return cls(divisions=divisions)
    
    def to_json_file(self, file_path: str) -> None:
        """Save FOR codes hierarchy to JSON file."""
        import json
        from pathlib import Path
        
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)


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

