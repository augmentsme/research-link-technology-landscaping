"""
Data models for PyRLA representing API entities
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class GeoPoint:
    """Represents a geographical point"""
    lat: float
    lon: float
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeoPoint':
        """Create GeoPoint from dictionary data"""
        return cls(
            lat=data.get('lat', 0.0),
            lon=data.get('lon', 0.0)
        )


@dataclass
class Vocab:
    """Represents vocabulary/subject classification"""
    label: str = ""
    type: str = ""
    version: str = ""
    notation: str = ""
    count: int = 0
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Vocab':
        """Create Vocab from dictionary data"""
        return cls(
            label=data.get('label', ''),
            type=data.get('type', ''),
            version=data.get('version', ''),
            notation=data.get('notation', ''),
            count=data.get('count', 0)
        )


@dataclass
class Summary:
    """Represents summary statistics and classifications"""
    organisation_count: int = 0
    grant_count: int = 0
    publication_count: int = 0
    patent_count: int = 0
    researcher_count: int = 0
    funding_amounts: List[float] = field(default_factory=list)
    lowest_funding: Optional[float] = None
    highest_funding: Optional[float] = None
    total_funding: Optional[float] = None
    vocabs: List[Vocab] = field(default_factory=list)
    primary_vocabs: List[Vocab] = field(default_factory=list)
    subjects: List[str] = field(default_factory=list)
    for_subjects: List[str] = field(default_factory=list)
    for_subject_codes: List[str] = field(default_factory=list)
    seo_subjects: List[str] = field(default_factory=list)
    seo_subject_codes: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Summary':
        """Create Summary from dictionary data"""
        if not data:
            return cls()
            
        return cls(
            organisation_count=data.get('organisationCount', 0),
            grant_count=data.get('grantCount', 0),
            publication_count=data.get('publicationCount', 0),
            patent_count=data.get('patentCount', 0),
            researcher_count=data.get('researcherCount', 0),
            funding_amounts=data.get('fundingAmounts', []),
            lowest_funding=data.get('lowestFunding'),
            highest_funding=data.get('highestFunding'),
            total_funding=data.get('totalFunding'),
            vocabs=[Vocab.from_dict(v) for v in (data.get('vocabs') or [])],
            primary_vocabs=[Vocab.from_dict(v) for v in (data.get('primaryVocabs') or [])],
            subjects=data.get('subjects', []),
            for_subjects=data.get('forSubjects', []),
            for_subject_codes=data.get('forSubjectCodes', []),
            seo_subjects=data.get('seoSubjects', []),
            seo_subject_codes=data.get('seoSubjectCodes', [])
        )


@dataclass
class Publication:
    """
    Represents a publication entity from the RLA API
    
    Based on RLA minimum metadata requirements for Publications:
    - Title: The publication title
    - Abstract: A brief summary of the publication's content
    - Publication Type: Journal article, conference paper, etc.
    - DOI: Digital Object Identifier providing permanent link
    - Publication Year: The year the publication was published
    """
    id: str = ""
    node_id: Optional[int] = None
    url: str = ""
    key: str = ""
    indexed_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    source: str = ""
    type: List[str] = field(default_factory=list)
    status: str = ""
    current: bool = True
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Core publication metadata (RLA minimum requirements)
    title: str = ""
    abstract: str = ""  # Changed from abstract_str for clarity
    publication_type: str = ""  # Journal article, conference paper, etc.
    doi: str = ""
    publication_year: Optional[int] = None
    publication_date: Optional[datetime] = None
    
    # Additional publication metadata
    publisher: str = ""
    authors_list: str = ""
    journal: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    isbn: str = ""
    issn: str = ""
    language: str = ""
    keywords: List[str] = field(default_factory=list)
    
    # Relationships
    researcher_ids: List[str] = field(default_factory=list)
    organisation_ids: List[str] = field(default_factory=list)
    grant_ids: List[str] = field(default_factory=list)
    
    # Geographic information
    countries: List[str] = field(default_factory=list)
    country_codes: List[str] = field(default_factory=list)
    cities: List[str] = field(default_factory=list)
    state_codes: List[str] = field(default_factory=list)
    states: List[str] = field(default_factory=list)
    suburbs: List[str] = field(default_factory=list)
    location: Optional[GeoPoint] = None
    locations: List[GeoPoint] = field(default_factory=list)
    summary: Optional[Summary] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Publication':
        """Create Publication from dictionary data"""
        # Extract publication year from date if available
        pub_year = None
        pub_date = cls._parse_datetime(data.get('publicationDate'))
        if pub_date:
            pub_year = pub_date.year
        elif data.get('publicationYear'):
            pub_year = int(data.get('publicationYear'))
        
        return cls(
            id=data.get('id', ''),
            node_id=data.get('nodeId'),
            url=data.get('url', ''),
            key=data.get('key', ''),
            indexed_date=cls._parse_datetime(data.get('indexedDate')),
            last_updated=cls._parse_datetime(data.get('lastUpdated')),
            source=data.get('source', ''),
            type=data.get('type', []),
            status=data.get('status', ''),
            current=data.get('current', True),
            start_date=cls._parse_datetime(data.get('startDate')),
            end_date=cls._parse_datetime(data.get('endDate')),
            
            # Core publication metadata
            title=data.get('title', ''),
            abstract=data.get('abstract', '') or data.get('abstractStr', ''),
            publication_type=data.get('publicationType', '') or data.get('type', [''])[0] if data.get('type') else '',
            doi=data.get('doi', ''),
            publication_year=pub_year,
            publication_date=pub_date,
            
            # Additional metadata
            publisher=data.get('publisher', ''),
            authors_list=data.get('authorsList', ''),
            journal=data.get('journal', ''),
            volume=data.get('volume', ''),
            issue=data.get('issue', ''),
            pages=data.get('pages', ''),
            isbn=data.get('isbn', ''),
            issn=data.get('issn', ''),
            language=data.get('language', ''),
            keywords=data.get('keywords', []),
            
            # Relationships
            researcher_ids=data.get('researcherIds', []),
            organisation_ids=data.get('organisationIds', []),
            grant_ids=data.get('grantIds', []),
            
            # Geographic data
            countries=data.get('countries', []),
            country_codes=data.get('countryCodes', []),
            cities=data.get('cities', []),
            state_codes=data.get('stateCodes', []),
            states=data.get('states', []),
            suburbs=data.get('suburbs', []),
            location=GeoPoint.from_dict(data['location']) if data.get('location') else None,
            locations=[GeoPoint.from_dict(loc) for loc in (data.get('locations') or [])],
            summary=Summary.from_dict(data.get('summary')) if data.get('summary') else None
        )
    
    @staticmethod
    def _parse_datetime(date_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO datetime string to datetime object"""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None


@dataclass
class Researcher:
    """Represents a researcher entity from the RLA API"""
    id: str = ""
    node_id: Optional[int] = None
    url: str = ""
    key: str = ""
    indexed_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    source: str = ""
    type: List[str] = field(default_factory=list)
    status: str = ""
    current: bool = True
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    address: str = ""
    countries: List[str] = field(default_factory=list)
    country_codes: List[str] = field(default_factory=list)
    cities: List[str] = field(default_factory=list)
    state_codes: List[str] = field(default_factory=list)
    states: List[str] = field(default_factory=list)
    suburbs: List[str] = field(default_factory=list)
    location: Optional[GeoPoint] = None
    locations: List[GeoPoint] = field(default_factory=list)
    full_name: str = ""
    first_name: str = ""
    last_name: str = ""
    orcid: str = ""
    scopus_author_id: str = ""
    publication_titles: List[str] = field(default_factory=list)
    summary: Optional[Summary] = None
    publications: List[Publication] = field(default_factory=list)
    current_organisation_name: List[str] = field(default_factory=list)
    grant_ids: List[str] = field(default_factory=list)
    organisation_ids: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Researcher':
        """Create Researcher from dictionary data"""
        return cls(
            id=data.get('id', ''),
            node_id=data.get('nodeId'),
            url=data.get('url', ''),
            key=data.get('key', ''),
            indexed_date=cls._parse_datetime(data.get('indexedDate')),
            last_updated=cls._parse_datetime(data.get('lastUpdated')),
            source=data.get('source', ''),
            type=data.get('type', []),
            status=data.get('status', ''),
            current=data.get('current', True),
            start_date=cls._parse_datetime(data.get('startDate')),
            end_date=cls._parse_datetime(data.get('endDate')),
            address=data.get('address', ''),
            countries=data.get('countries', []),
            country_codes=data.get('countryCodes', []),
            cities=data.get('cities', []),
            state_codes=data.get('stateCodes', []),
            states=data.get('states', []),
            suburbs=data.get('suburbs', []),
            location=GeoPoint.from_dict(data['location']) if data.get('location') else None,
            locations=[GeoPoint.from_dict(loc) for loc in (data.get('locations') or [])],
            full_name=data.get('fullName', ''),
            first_name=data.get('firstName', ''),
            last_name=data.get('lastName', ''),
            orcid=data.get('orcid', ''),
            scopus_author_id=data.get('scopusAuthorId', ''),
            publication_titles=data.get('publicationTitles', []),
            summary=Summary.from_dict(data.get('summary')) if data.get('summary') else None,
            publications=[Publication.from_dict(pub) for pub in (data.get('publications') or [])],
            current_organisation_name=data.get('currentOrganisationName', []),
            grant_ids=data.get('grantIds', []),
            organisation_ids=data.get('organisationIds', [])
        )
    
    @staticmethod
    def _parse_datetime(date_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO datetime string to datetime object"""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None
    
    def __str__(self) -> str:
        """String representation of researcher"""
        return f"Researcher(id='{self.id}', name='{self.full_name}', orcid='{self.orcid}')"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return self.__str__()


@dataclass
class Grant:
    """Represents a grant entity from the RLA API"""
    id: str = ""
    node_id: Optional[int] = None
    url: str = ""
    key: str = ""
    indexed_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    source: str = ""
    type: List[str] = field(default_factory=list)
    status: str = ""
    current: bool = True
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    address: str = ""
    countries: List[str] = field(default_factory=list)
    country_codes: List[str] = field(default_factory=list)
    cities: List[str] = field(default_factory=list)
    state_codes: List[str] = field(default_factory=list)
    states: List[str] = field(default_factory=list)
    suburbs: List[str] = field(default_factory=list)
    location: Optional[GeoPoint] = None
    locations: List[GeoPoint] = field(default_factory=list)
    title: str = ""
    grant_title: str = ""
    funder: str = ""
    grant_id: str = ""
    funding_scheme: str = ""
    funding_amount: Optional[float] = None
    participant_list: str = ""
    grant_summary: str = ""
    grant_statement: str = ""
    admin_organisation_name: str = ""
    admin_organisation: str = ""
    primary_subject: str = ""
    summary: Optional[Summary] = None
    researcher_ids: List[str] = field(default_factory=list)
    organisation_ids: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Grant':
        """Create Grant from dictionary data"""
        return cls(
            id=data.get('id', ''),
            node_id=data.get('nodeId'),
            url=data.get('url', ''),
            key=data.get('key', ''),
            indexed_date=cls._parse_datetime(data.get('indexedDate')),
            last_updated=cls._parse_datetime(data.get('lastUpdated')),
            source=data.get('source', ''),
            type=data.get('type', []),
            status=data.get('status', ''),
            current=data.get('current', True),
            start_date=cls._parse_datetime(data.get('startDate')),
            end_date=cls._parse_datetime(data.get('endDate')),
            address=data.get('address', ''),
            countries=data.get('countries', []),
            country_codes=data.get('countryCodes', []),
            cities=data.get('cities', []),
            state_codes=data.get('stateCodes', []),
            states=data.get('states', []),
            suburbs=data.get('suburbs', []),
            location=GeoPoint.from_dict(data['location']) if data.get('location') else None,
            locations=[GeoPoint.from_dict(loc) for loc in (data.get('locations') or [])],
            title=data.get('title', ''),
            grant_title=data.get('grantTitle', ''),
            funder=data.get('funder', ''),
            grant_id=data.get('grantId', ''),
            funding_scheme=data.get('fundingScheme', ''),
            funding_amount=data.get('fundingAmount'),
            participant_list=data.get('participantList', ''),
            grant_summary=data.get('grantSummary', ''),
            grant_statement=data.get('grantStatement', ''),
            admin_organisation_name=data.get('adminOrganisationName', ''),
            admin_organisation=data.get('adminOrganisation', ''),
            primary_subject=data.get('primarySubject', ''),
            summary=Summary.from_dict(data.get('summary')) if data.get('summary') else None,
            researcher_ids=data.get('researcherIds', []),
            organisation_ids=data.get('organisationIds', [])
        )
    
    @staticmethod
    def _parse_datetime(date_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO datetime string to datetime object"""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None
    
    def __str__(self) -> str:
        """String representation of grant"""
        title = self.grant_title or self.title
        return f"Grant(id='{self.id}', title='{title[:50]}...', funder='{self.funder}')"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return self.__str__()


@dataclass
class Organisation:
    """Represents an organisation entity from the RLA API"""
    id: str = ""
    node_id: Optional[int] = None
    url: str = ""
    key: str = ""
    indexed_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    source: str = ""
    type: List[str] = field(default_factory=list)
    status: str = ""
    current: bool = True
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    address: str = ""
    countries: List[str] = field(default_factory=list)
    country_codes: List[str] = field(default_factory=list)
    cities: List[str] = field(default_factory=list)
    state_codes: List[str] = field(default_factory=list)
    states: List[str] = field(default_factory=list)
    suburbs: List[str] = field(default_factory=list)
    location: Optional[GeoPoint] = None
    locations: List[GeoPoint] = field(default_factory=list)
    name: str = ""
    unique_name: str = ""
    organisation_name: str = ""
    doi: str = ""
    abn: str = ""
    abr_type: str = ""
    abr_type_raw: str = ""
    anzsic_label: str = ""
    anzsic_label_raw: str = ""
    anzsic: str = ""
    ror: str = ""
    abn_cancellation_date: Optional[datetime] = None
    summary: Optional[Summary] = None
    researcher_ids: List[str] = field(default_factory=list)
    grant_ids: List[str] = field(default_factory=list)
    organisation_ids: List[str] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Organisation':
        """Create Organisation from dictionary data"""
        return cls(
            id=data.get('id', ''),
            node_id=data.get('nodeId'),
            url=data.get('url', ''),
            key=data.get('key', ''),
            indexed_date=cls._parse_datetime(data.get('indexedDate')),
            last_updated=cls._parse_datetime(data.get('lastUpdated')),
            source=data.get('source', ''),
            type=data.get('type', []),
            status=data.get('status', ''),
            current=data.get('current', True),
            start_date=cls._parse_datetime(data.get('startDate')),
            end_date=cls._parse_datetime(data.get('endDate')),
            address=data.get('address', ''),
            countries=data.get('countries', []),
            country_codes=data.get('countryCodes', []),
            cities=data.get('cities', []),
            state_codes=data.get('stateCodes', []),
            states=data.get('states', []),
            suburbs=data.get('suburbs', []),
            location=GeoPoint.from_dict(data['location']) if data.get('location') else None,
            locations=[GeoPoint.from_dict(loc) for loc in (data.get('locations') or [])],
            name=data.get('name', ''),
            unique_name=data.get('uniqueName', ''),
            organisation_name=data.get('organisationName', ''),
            doi=data.get('doi', ''),
            abn=data.get('abn', ''),
            abr_type=data.get('abrType', ''),
            abr_type_raw=data.get('abrTypeRaw', ''),
            anzsic_label=data.get('anzsicLabel', ''),
            anzsic_label_raw=data.get('anzsicLabelRaw', ''),
            anzsic=data.get('anzsic', ''),
            ror=data.get('ror', ''),
            abn_cancellation_date=cls._parse_datetime(data.get('abnCanellationDate')),
            summary=Summary.from_dict(data.get('summary')) if data.get('summary') else None,
            researcher_ids=data.get('researcherIds', []),
            grant_ids=data.get('grantIds', []),
            organisation_ids=data.get('organisationIds', [])
        )
    
    @staticmethod
    def _parse_datetime(date_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO datetime string to datetime object"""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return None
    
    def __str__(self) -> str:
        """String representation of organisation"""
        name = self.organisation_name or self.name
        return f"Organisation(id='{self.id}', name='{name}', abn='{self.abn}')"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return self.__str__()


@dataclass 
class SearchResponse:
    """Generic search response wrapper"""
    total_results: int = 0
    current_page: int = 1
    from_index: int = 0
    size: int = 10
    results: List[Any] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], result_class=None) -> 'SearchResponse':
        """Create SearchResponse from dictionary data"""
        results = []
        if result_class and data.get('results'):
            results = [result_class.from_dict(item) for item in data['results']]
        else:
            results = data.get('results', [])
            
        return cls(
            total_results=data.get('totalResults', 0),
            current_page=data.get('currentPage', 1),
            from_index=data.get('from', 0),
            size=data.get('size', 10),
            results=results
        )
