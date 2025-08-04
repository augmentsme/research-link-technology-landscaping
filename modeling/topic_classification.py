import json
import os

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import multiple_choice


class ClassificationType(Enum):
    """Enum for different classification types."""
    DOMAINS = "domains"
    FIELDS = "fields"
    SUBFIELDS = "subfields"
    TOPICS = "topics"


class ClassificationTemplate(str, Enum):
    """Templates for classification questions using OpenAlex IDs."""
    
    SINGLE_ANSWER = """Answer the following multiple choice question. The entire content of your response should be of the following format: 'ANSWER: $ID' (without quotes) where ID is one of the OpenAlex IDs from the choices below.

{question}

{choices}

Available IDs: {ids}
"""
    
    SINGLE_ANSWER_COT = """Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $ID' (without quotes) where ID is one of the OpenAlex IDs from the choices below. Think step by step before answering.

{question}

{choices}

Available IDs: {ids}
"""
    
    MULTIPLE_ANSWER = """Answer the following multiple choice question where multiple answers may be correct. The entire content of your response should be of the following format: 'ANSWER: $IDS' (without quotes) where IDS is one or more OpenAlex IDs from the choices below, separated by commas.

{question}

{choices}

Available IDs: {ids}
"""
    
    MULTIPLE_ANSWER_COT = """Answer the following multiple choice question where multiple answers may be correct. The last line of your response should be of the following format: 'ANSWER: $IDS' (without quotes) where IDS is one or more OpenAlex IDs from the choices below, separated by commas. Think step by step before answering.

{question}

{choices}

Available IDs: {ids}
"""


def extract_canonical_id(openalex_id: str) -> str:
    """Extract canonical ID from OpenAlex URL or return as-is if already canonical."""
    if openalex_id.startswith('https://openalex.org/'):
        return openalex_id.split('/')[-1]
    return openalex_id

@dataclass
class Label:
    """Represents a classification label (domain, field, subfield, or topic)."""
    id: str
    display_name: str
    description: str
    keywords: List[str] = None
    field: str = None
    domain: str = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], classification_type: ClassificationType) -> 'Label':
        """Create Label from OpenAlex data dictionary."""
        if classification_type == ClassificationType.TOPICS:
            return cls(
                id=data.get('id', ''),
                display_name=data.get('display_name', ''),
                description=data.get('description', ''),
                keywords=data.get('keywords', []),
                field=data.get('field', {}).get('display_name', ''),
                domain=data.get('domain', {}).get('display_name', '')
            )
        elif classification_type == ClassificationType.DOMAINS:
            return cls(
                id=data.get('id', ''),
                display_name=data.get('display_name', ''),
                description=data.get('description', ''),
                keywords=None,
                field=None,
                domain=None
            )
        elif classification_type == ClassificationType.FIELDS:
            return cls(
                id=data.get('id', ''),
                display_name=data.get('display_name', ''),
                description=data.get('description', ''),
                keywords=None,
                field=None,
                domain=data.get('domain', {}).get('display_name', '') if data.get('domain') else None
            )
        elif classification_type == ClassificationType.SUBFIELDS:
            return cls(
                id=data.get('id', ''),
                display_name=data.get('display_name', ''),
                description=data.get('description', ''),
                keywords=None,
                field=data.get('field', {}).get('display_name', '') if data.get('field') else None,
                domain=data.get('domain', {}).get('display_name', '') if data.get('domain') else None
            )


@dataclass
class Topic(Label):
    """Represents an OpenAlex research topic (legacy compatibility)."""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Topic':
        """Create Topic from OpenAlex topic dictionary."""
        return cls(
            id=data.get('id', ''),
            display_name=data.get('display_name', ''),
            description=data.get('description', ''),
            keywords=data.get('keywords', []),
            field=data.get('field', {}).get('display_name', ''),
            domain=data.get('domain', {}).get('display_name', '')
        )


@dataclass
class Grant:
    """Represents a research grant with its metadata."""
    id: str
    title: str
    summary: str
    funding_amount: Optional[float] = None
    
    @property
    def has_summary(self) -> bool:
        """Check if grant has a non-empty summary."""
        return bool(self.summary and self.summary.strip())
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Grant':
        """Create Grant from dictionary data."""
        return cls(
            id=data.get('id', ''),
            title=data.get('title', ''),
            summary=data.get('grant_summary', ''),
            funding_amount=data.get('funding_amount', None)
        )


@dataclass
class PipelineConfig:
    """Configuration for the research pipeline."""
    data_path: str = "/home/lcheng/oz318/research-link-technology-landscaping/data/active_grants.json"
    base_data_dir: str = "/home/lcheng/oz318/research-link-technology-landscaping/data"
    classification_type: ClassificationType = ClassificationType.TOPICS
    max_grants_classification: Optional[int] = None  # None means use all grants
    max_labels_for_classification: Optional[int] = None  # None means use all labels
    log_dir: str = "pipeline_logs"
    labels: List[Label] = None
    
    def __post_init__(self):
        """Load labels from the appropriate JSON file based on classification type."""
        if self.labels is None:
            self.labels = self._load_labels_from_file()
    
    @property
    def labels_path(self) -> str:
        """Get the path to the labels file based on classification type."""
        filename = f"{self.classification_type.value}.json"
        return os.path.join(self.base_data_dir, filename)
    
    @property 
    def topics_path(self) -> str:
        """Legacy property for backward compatibility."""
        return os.path.join(self.base_data_dir, "topics.json")
    
    @property
    def research_topics(self) -> List[Topic]:
        """Legacy property for backward compatibility - returns labels as Topics."""
        if self.classification_type == ClassificationType.TOPICS:
            return [Topic(
                id=label.id,
                display_name=label.display_name,
                description=label.description,
                keywords=label.keywords or [],
                field=label.field or '',
                domain=label.domain or ''
            ) for label in self.labels]
        else:
            # For non-topic classifications, convert to Topic format for compatibility
            return [Topic(
                id=label.id,
                display_name=label.display_name,
                description=label.description,
                keywords=[],
                field=label.field or '',
                domain=label.domain or ''
            ) for label in self.labels]
    
    def _load_labels_from_file(self) -> List[Label]:
        """Load labels from the appropriate JSON file based on classification type."""
        labels_path = self.labels_path
        with open(labels_path, 'r') as f:
            labels_data = json.load(f)
        
        # Convert to Label objects and limit the number if specified
        labels = [Label.from_dict(label_data, self.classification_type) for label_data in labels_data]
        
        if self.max_labels_for_classification is not None:
            # Take the first N labels or could be filtered by domain/field if needed
            labels = labels[:self.max_labels_for_classification]
            
        return labels




class DataLoader:
    """Handles loading and processing of grant data."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def load_grants(self) -> List[Grant]:
        """Load grants from available data sources."""
        data_path = self.config.data_path
        with open(data_path, 'r') as f:
            raw_grants = json.load(f)
        
        grants = []
        for grant_data in raw_grants:
            grant = Grant.from_dict(grant_data)
            if grant.has_summary:
                grants.append(grant)
        
        return grants
    
    def load_labels(self) -> List[Label]:
        """Load labels from the appropriate file based on classification type."""
        return self.config.labels
    
    def load_topics(self) -> List[Topic]:
        """Load topics from OpenAlex topics file (legacy compatibility)."""
        return self.config.research_topics
    





class InspectAITaskCreator:
    """Creates Inspect AI tasks for evaluation."""
    
    def __init__(self, config: PipelineConfig, grants: List[Grant]):
        self.config = config
        self.grants = grants
        self.labels = config.labels
        self.topics = config.research_topics  # For backward compatibility
    
    def _get_user_template(self) -> str:
        """Get the user message template for classification."""
        return """# Grant Classification Task

## Available Research {label_type}

{topics_str}

## Grant to Classify

**Title:** {grant_title}
**Summary:** {grant_summary}

Please classify this grant into the most appropriate OpenAlex research {label_type} from the list above.

"""
    
    def _format_labels_for_classification(self) -> str:
        """Format labels for classification task based on classification type."""
        formatted_labels = []
        
        for i, label in enumerate(self.labels, 1):
            if self.config.classification_type == ClassificationType.TOPICS:
                # For topics, include keywords and field/domain info
                keywords_str = ", ".join(label.keywords[:5]) if label.keywords else "N/A"
                label_text = f"""{i}. ID: {label.id}
   Topic: {label.display_name}
   Field: {label.field} | Domain: {label.domain}
   Description: {label.description}
   Key Keywords: {keywords_str}
"""
            elif self.config.classification_type == ClassificationType.DOMAINS:
                # For domains, simpler format
                label_text = f"""{i}. ID: {label.id}
   Domain: {label.display_name}
   Description: {label.description}
"""
            elif self.config.classification_type == ClassificationType.FIELDS:
                # For fields, include domain if available
                domain_str = f" | Domain: {label.domain}" if label.domain else ""
                label_text = f"""{i}. ID: {label.id}
   Field: {label.display_name}{domain_str}
   Description: {label.description}
"""
            elif self.config.classification_type == ClassificationType.SUBFIELDS:
                # For subfields, include field and domain if available
                field_str = f" | Field: {label.field}" if label.field else ""
                domain_str = f" | Domain: {label.domain}" if label.domain else ""
                label_text = f"""{i}. ID: {label.id}
   Subfield: {label.display_name}{field_str}{domain_str}
   Description: {label.description}
"""
            
            formatted_labels.append(label_text)
        
        return "\n".join(formatted_labels)
    
    def _format_topics_for_classification(self) -> str:
        """Format OpenAlex topics for classification task (legacy compatibility)."""
        return self._format_labels_for_classification()
    
    def _concatenate_summaries(self, grants: List[Grant]) -> str:
        """Concatenate grant summaries into a single text."""
        summaries = [f"Grant {i+1}: {grant.summary}" for i, grant in enumerate(grants)]
        return "\n\n".join(summaries)
    
    def create_classification_prompt(self, grant: Grant) -> str:
        """Create a classification prompt for a specific grant using the user template."""
        user_template = self._get_user_template()
        labels_str = self._format_labels_for_classification()
        
        # Get the appropriate label type name for the prompt
        label_type_name = {
            ClassificationType.DOMAINS: "domains",
            ClassificationType.FIELDS: "fields", 
            ClassificationType.SUBFIELDS: "subfields",
            ClassificationType.TOPICS: "topics"
        }[self.config.classification_type]
        
        return user_template.format(
            topics_str=labels_str,  # Keep the same placeholder for compatibility
            grant_title=grant.title,
            grant_summary=grant.summary,
            label_type=label_type_name,
            classification_type=self.config.classification_type.value
        )
    
    def create_classification_prompt_for_multiple_choice(self, grant: Grant) -> str:
        """Create a simplified classification prompt for multiple choice format."""
        # Get the appropriate label type name for the prompt
        label_type_name = {
            ClassificationType.DOMAINS: "domain",
            ClassificationType.FIELDS: "field", 
            ClassificationType.SUBFIELDS: "subfield",
            ClassificationType.TOPICS: "topic"
        }[self.config.classification_type]
        
        prompt = f"""# Grant Classification Task

## Grant to Classify

**Title:** {grant.title}
**Summary:** {grant.summary}

Please classify this grant into the most appropriate OpenAlex research {label_type_name} from the available choices.

Select the {label_type_name} that best matches the research focus and content of this grant.

Respond with the canonical OpenAlex ID (e.g. "1" for domains, "23" for fields, "2403" for subfields and "T11881" for topics) of your chosen {label_type_name}."""
        
        return prompt


def get_classification_template(cot: bool = False, multiple_correct: bool = False) -> ClassificationTemplate:
    """
    Convenience function to get the appropriate template based on options.
    
    Args:
        cot: Whether to use chain-of-thought reasoning
        multiple_correct: Whether multiple answers are allowed
        
    Returns:
        The appropriate ClassificationTemplate
    """
    if multiple_correct:
        return ClassificationTemplate.MULTIPLE_ANSWER_COT if cot else ClassificationTemplate.MULTIPLE_ANSWER
    else:
        return ClassificationTemplate.SINGLE_ANSWER_COT if cot else ClassificationTemplate.SINGLE_ANSWER


@task
def classify_grants_task(
    classification_type: Union[ClassificationType, str] = ClassificationType.SUBFIELDS, 
    max_grants: Optional[Union[int, str]] = None, 
    max_labels: Optional[Union[int, str]] = None,
    cot: bool = False,
    multiple_correct: bool = False
) -> Task:
    """
    Task for classifying grants into OpenAlex classifications.
    
    Args:
        classification_type: Type of classification (domains, fields, subfields, topics)
        max_grants: Maximum number of grants to include (None for all)
        max_labels: Maximum number of labels to include (None for all)
        cot: Whether to use chain-of-thought reasoning
        multiple_correct: Whether multiple answers are allowed
    """
    if isinstance(classification_type, str):
        classification_type = ClassificationType(classification_type.lower())
    
    # Convert string parameters to integers if needed
    if isinstance(max_grants, str):
        try:
            max_grants = int(max_grants)
        except ValueError:
            raise ValueError(f"max_grants must be an integer or None, got: {max_grants}")
    
    if isinstance(max_labels, str):
        try:
            max_labels = int(max_labels)
        except ValueError:
            raise ValueError(f"max_labels must be an integer or None, got: {max_labels}")
    
    # Determine template from cot and multiple_correct parameters
    template = get_classification_template(cot=cot, multiple_correct=multiple_correct)
    
    config = PipelineConfig(
        classification_type=classification_type,
        max_grants_classification=max_grants,
        max_labels_for_classification=max_labels
    )
    
    # Load data
    data_loader = DataLoader(config)
    grants = data_loader.load_grants()
    
    # Apply limits
    if config.max_grants_classification is not None:
        grants_to_use = grants[:config.max_grants_classification]
    else:
        grants_to_use = grants
    
    # Create task creator to handle prompt generation
    task_creator = InspectAITaskCreator(config, grants_to_use)
    
    # Create the template with available IDs
    available_ids = ', '.join([extract_canonical_id(label.id) for label in config.labels])
    custom_template = template.value.format(
        question="{question}",
        choices="{choices}", 
        ids=available_ids
    )
    
    samples = []
    for grant in grants_to_use:
        # Create choices from available labels using canonical IDs as option labels
        choices = []
        for label in config.labels:
            canonical_id = extract_canonical_id(label.id)
            choice_text = f"<{canonical_id}> {label.display_name}: {label.description}"
            choices.append(choice_text)
        
        # Use the task creator to generate the user prompt (simplified for multiple choice)
        user_input = task_creator.create_classification_prompt_for_multiple_choice(grant)

        sample = Sample(
            input=user_input,
            choices=choices,
            metadata={
                "grant_id": grant.id,
                "grant_title": grant.title,
                "grant_summary": grant.summary,
                "num_labels": len(config.labels),
                "classification_type": classification_type.value,
                "label_ids": [extract_canonical_id(label.id) for label in config.labels],
                "label_names": [label.display_name for label in config.labels],
                "solver_type": "multiple_choice",
                "template_type": template.name
            }
        )
        samples.append(sample)
    
    return Task(
        dataset=samples,
        solver=multiple_choice(template=custom_template)
    )