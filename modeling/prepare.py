import json
import asyncio
import concurrent.futures
from neo4j import GraphDatabase
import os
import sys
from pathlib import Path
import pandas as pd
from config import DATA_DIR
import config
import utils
from typing import List, Dict, Any, Optional, Union
from for_codes_cleaner import main as clean_for_codes
import httpx
import logging
from urllib.parse import urljoin
from datetime import datetime
from dataclasses import dataclass, field
import typer


def get_cipher_query(limit: Optional[int] = None) -> str:
    """Generate Neo4j cipher query with optional limit"""
    base_query = "MATCH (g:grant) RETURN g"
    if limit is not None:
        return f"{base_query} LIMIT {limit}"
    return base_query


# RLA Client Configuration
class APIConfig:
    """API-related configuration constants"""
    DEFAULT_TIMEOUT = 60
    DEFAULT_PAGE_SIZE = 10
    MAX_PAGE_SIZE = 250
    DEFAULT_MAX_CONCURRENT = 100
    BASE_URL = "https://researchlink.ardc.edu.au"


# RLA Client Exceptions
class RLAError(Exception):
    """Base exception class for RLA operations"""
    pass


class RLAAuthenticationError(RLAError):
    """Raised when authentication fails"""
    pass


class RLANotFoundError(RLAError):
    """Raised when a resource is not found"""
    pass


class RLAValidationError(RLAError):
    """Raised when request validation fails"""
    pass


class RLAServerError(RLAError):
    """Raised when server error occurs"""
    pass


# RLA Data Models
@dataclass
class Grant:
    """Represents a grant entity from the RLA API"""
    id: str = ""
    key: str = ""
    title: str = ""
    grant_title: str = ""
    grant_summary: str = ""
    funder: str = ""
    grant_id: str = ""
    funding_amount: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Grant':
        """Create Grant from dictionary data"""
        return cls(
            id=data.get('id', ''),
            key=data.get('key', ''),
            title=data.get('title', ''),
            grant_title=data.get('grantTitle', ''),
            grant_summary=data.get('grantSummary', ''),
            funder=data.get('funder', ''),
            grant_id=data.get('grantId', ''),
            funding_amount=data.get('fundingAmount')
        )


@dataclass
class SearchResponse:
    """Response object for search operations"""
    total_results: int = 0
    current_page: int = 1
    from_index: int = 0
    size: int = 0
    results: List[Grant] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], result_type) -> 'SearchResponse':
        """Create SearchResponse from dictionary data"""
        results = []
        if 'results' in data and isinstance(data['results'], list):
            results = [result_type.from_dict(item) for item in data['results']]
        
        return cls(
            total_results=data.get('totalResults', 0),
            current_page=data.get('currentPage', 1),
            from_index=data.get('fromIndex', 0),
            size=data.get('size', 0),
            results=results
        )


class RLAClient:
    """Real RLA client implementation for grant enrichment"""
    
    def __init__(self, api_token: Optional[str] = None, base_url: str = APIConfig.BASE_URL, debug: bool = False):
        self.base_url = base_url.rstrip('/')
        self.debug = debug or os.getenv('PYRLA_DEBUG', '').lower() in ('1', 'true', 'yes')
        
        # Get token from parameter or environment
        self.api_token = api_token or os.getenv('ARL_API_TOKEN')
        if not self.api_token:
            raise RLAAuthenticationError(
                "API token is required. Provide it via api_token parameter or ARL_API_TOKEN environment variable"
            )
        
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        self.timeout = APIConfig.DEFAULT_TIMEOUT
        self._session: Optional[httpx.AsyncClient] = None
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
    
    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create async session"""
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient(
                headers=self.headers,
                timeout=self.timeout
            )
        return self._session
    
    async def _close_session(self):
        """Close async session"""
        if self._session and not self._session.is_closed:
            await self._session.aclose()
            self._session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._close_session()
    
    async def _make_async_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make async HTTP request to API endpoint"""
        url = urljoin(self.base_url, endpoint)
        session = await self._get_session()
        
        try:
            if self.debug:
                if params:
                    param_str = "&".join([f"{k}={v}" for k, v in (params or {}).items()])
                    full_url = f"{url}?{param_str}" if param_str else url
                    print(f"[DEBUG] API Request: {full_url}")
                else:
                    print(f"[DEBUG] API Request: {url}")
                
            response = await session.get(url, params=params or {})
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                raise RLAAuthenticationError("Authentication failed - check API token")
            elif response.status_code == 404:
                raise RLANotFoundError("Resource not found")
            elif response.status_code == 400:
                raise RLAValidationError(f"Invalid request parameters: {response.text}")
            elif 500 <= response.status_code < 600:
                raise RLAServerError(f"Server error ({response.status_code}): {response.text}")
            else:
                raise RLAError(f"API request failed ({response.status_code}): {response.text}")
                    
        except httpx.RequestError as e:
            raise RLAError(f"Request failed: {e}")
    
    async def search_grants(
        self,
        value: str = "",
        filter_query: str = "",
        page_number: int = 1,
        page_size: int = APIConfig.DEFAULT_PAGE_SIZE,
        advanced_search_query: str = ""
    ) -> SearchResponse:
        """Search for grants"""
        params = {
            "value": value,
            "filterQuery": filter_query,
            "pageNumber": page_number,
            "pageSize": min(page_size, APIConfig.MAX_PAGE_SIZE),
            "advancedSearchQuery": advanced_search_query
        }
        
        # Remove empty parameters
        params = {k: v for k, v in params.items() if v}
        
        data = await self._make_async_request("/search/grants", params)
        return SearchResponse.from_dict(data, Grant)
    
    async def get_grant(self, grant_id: str) -> Grant:
        """Get a specific grant by ID"""
        data = await self._make_async_request(f"/search/grant/{grant_id}")
        return Grant.from_dict(data)


class GrantsEnricher:
    """Handles grant data enrichment with summaries"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.logger = logging.getLogger(__name__)
    
    async def enrich_grants_with_summaries(self, client: RLAClient, input_records: List[dict], key: str = "key") -> List[dict]:
        """Enrich grant records with grant summaries from the RLA API"""
        enriched_records = []
        semaphore = asyncio.Semaphore(APIConfig.DEFAULT_MAX_CONCURRENT)
        
        async def enrich_single_record(record: dict, index: int):
            async with semaphore:
                try:
                    # Only use key-based search strategy
                    grant = None
                    
                    # Search by key if it exists
                    if key in record and record[key]:
                        key_value = record[key]
                        
                        try:
                            # Search by key filter
                            response = await client.search_grants(
                                value="",
                                filter_query=f"key:{key_value}",
                                page_size=1
                            )
                            
                            if response.results:
                                grant = response.results[0]
                            else:
                                # Try extracting ID from key (e.g., "nhmrc/1047630" -> "1047630") as fallback
                                if '/' in key_value:
                                    extracted_id = key_value.split('/')[-1]
                                    try:
                                        grant = await client.get_grant(extracted_id)
                                    except RLAError:
                                        if self.debug:
                                            print(f"Failed to get grant by extracted ID '{extracted_id}' from key '{key_value}'")
                                
                        except RLAError as e:
                            if self.debug:
                                print(f"Search by key '{key_value}' failed: {e}")
                    else:
                        if self.debug:
                            print(f"Record {index + 1}: No '{key}' field found in record")
                    
                    # Create enriched record
                    enriched_record = record.copy()
                    if grant and grant.grant_summary:
                        enriched_record['grant_summary'] = grant.grant_summary
                        if self.debug:
                            print(f"Enriched record {index + 1}: Found grant summary")
                    else:
                        enriched_record['grant_summary'] = ""
                        if self.debug:
                            print(f"Record {index + 1}: No grant summary found for {key} '{record.get(key, f'No {key} found')}'")
                    
                    return enriched_record
                    
                except Exception as e:
                    if self.debug:
                        print(f"Error processing record {index + 1}: {e}")
                    enriched_record = record.copy()
                    enriched_record['grant_summary'] = ""
                    return enriched_record
        
        # Process all records concurrently
        tasks = [enrich_single_record(record, i) for i, record in enumerate(input_records)]
        enriched_records = await asyncio.gather(*tasks, return_exceptions=False)
        
        return enriched_records
    
    async def enrich_grants_data(self, grants_data: List[dict]) -> List[dict]:
        """Enrich grants data with summaries from the RLA API"""
        if not grants_data:
            return grants_data
        
        try:
            client = RLAClient(debug=self.debug)
            async with client:
                enriched_data = await self.enrich_grants_with_summaries(
                    client=client,
                    input_records=grants_data,
                    key="key"
                )
                return enriched_data
                
        except Exception as e:
            print(f"Error during enrichment: {e}")
            # Return data with empty summaries as fallback
            fallback_data = []
            for record in grants_data:
                enriched_record = record.copy()
                enriched_record['grant_summary'] = ""
                fallback_data.append(enriched_record)
            return fallback_data
    
    def run_async_enrichment(self, grants_data: List[dict]) -> List[dict]:
        """Helper function to run async enrichment in sync context"""
        try:
            # Check if we're already in an event loop
            try:
                asyncio.get_running_loop()
                # If we're already in an event loop, create a new one in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.enrich_grants_data(grants_data))
                    return future.result()
            except RuntimeError:
                # No event loop is running, safe to use asyncio.run
                return asyncio.run(self.enrich_grants_data(grants_data))
        except Exception:
            # Fallback: create new event loop
            return asyncio.run(self.enrich_grants_data(grants_data))


class Neo4jConnector:
    """Handles Neo4j database connections and queries"""
    
    def __init__(self, uri: str, username: str, password: str):
        self.uri = uri
        self.username = username
        self.password = password
    
    def get_grant_nodes_as_json(self, query: str) -> Optional[List[dict]]:
        """
        Connects to Neo4j, fetches nodes with the 'grant' label,
        and returns them as a list of dictionaries.
        """
        try:
            with GraphDatabase.driver(self.uri, auth=(self.username, self.password)) as driver:
                driver.verify_connectivity()
                records, summary, keys = driver.execute_query(query, database_="neo4j")
                
                grant_nodes = []
                for record in records:
                    # Each record contains a node 'g'.
                    node = record["g"]
                    # Convert the node properties to a dictionary.
                    grant_nodes.append(dict(node))
                
                return grant_nodes
        
        except Exception as e:
            print(f"Error fetching grants from Neo4j: {e}")
            return None


class GrantsCleaner:
    """
    Handles cleaning and processing of enriched grant data.
    
    Includes deduplication across sources while preserving detailed information
    from authoritative sources (NHMRC/ARC) and merging organizational links from ORCID.
    """
    
    def clean_enriched_data(self, enriched_file_path: Path) -> Optional[pd.DataFrame]:
        """
        Clean the enriched grants data by filtering and standardizing data from all sources:
        - nhmrc.org: NHMRC grants with funding amounts
        - arc.gov.au: ARC grants with funding amounts and FOR codes
        - orcid.org: Additional grant records (often without summaries)
        - crossref.org: International grants with funding amounts
        
        Args:
            enriched_file_path: Path to the enriched grants JSONL file
            
        Returns:
            pandas.DataFrame: Cleaned grants data with standardized columns
        """
        try:
            df = utils.load_jsonl_file(enriched_file_path, as_dataframe=True)
            df = df.rename(columns={"key": "id"})
            df = df.set_index("id")
            
            print(f"Loaded {len(df):,} total records from {len(df.source.unique())} sources")
            for source in df.source.unique():
                count = len(df[df.source == source])
                print(f"  - {source}: {count:,} records")
            
            return self._clean_all_sources(df)
        
        except Exception as e:
            print(f"Error during data cleaning: {e}")
            return None
    
    def _clean_all_sources(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean grant data from all sources with unified approach"""
        cleaned_parts = []
        
        # Process NHMRC grants (nhmrc.org)
        nhmrc_data = self._clean_nhmrc_grants(df[df.source == 'nhmrc.org'])
        if not nhmrc_data.empty:
            cleaned_parts.append(nhmrc_data)
            print(f"Processed NHMRC grants: {len(nhmrc_data):,} records")
        
        # Process ARC grants (arc.gov.au)
        arc_data = self._clean_arc_grants(df[df.source == 'arc.gov.au'])
        if not arc_data.empty:
            cleaned_parts.append(arc_data)
            print(f"Processed ARC grants: {len(arc_data):,} records")
        
        # Process ORCID grants (orcid.org) - filter for Australian funders
        orcid_data = self._clean_orcid_grants(df[df.source == 'orcid.org'])
        if not orcid_data.empty:
            cleaned_parts.append(orcid_data)
            print(f"Processed ORCID grants: {len(orcid_data):,} records")
        
        # Process Crossref grants (crossref.org) - keep all with funding amounts
        crossref_data = self._clean_crossref_grants(df[df.source == 'crossref.org'])
        if not crossref_data.empty:
            cleaned_parts.append(crossref_data)
            print(f"Processed Crossref grants: {len(crossref_data):,} records")
            
        df_cleaned = pd.concat(cleaned_parts, axis=0, ignore_index=False)
        df_cleaned = self._standardize_final_columns(df_cleaned)
        
        normalized_titles = df_cleaned.title.apply(utils.normalize)
        
        # Deduplicate grants with same normalized titles
        # Priority: arc > nhmrc > orcid > crossref
        # Within same source: prefer records with valid grant_summary
        
        # Add normalized title column
        df_cleaned['normalized_title'] = normalized_titles
        
        # Create source priority mapping (lower number = higher priority)
        source_priority = {
            'arc.gov.au': 1,
            'nhmrc.org': 2, 
            'orcid.org': 3,
            'crossref.org': 4
        }
        df_cleaned['source_priority'] = df_cleaned['source'].map(source_priority)
        
        # Create grant_summary validity indicator (1 for valid, 0 for empty/NA)
        df_cleaned['has_summary'] = (
            df_cleaned['grant_summary'].notna() & 
            (df_cleaned['grant_summary'].str.strip() != '')
        ).astype(int)
        
        # Sort by normalized_title, then source_priority (ascending), then has_summary (descending)
        df_cleaned = df_cleaned.sort_values([
            'normalized_title', 
            'source_priority', 
            'has_summary'
        ], ascending=[True, True, False])
        
        # Keep only the first record for each normalized title (highest priority)
        df_deduplicated = df_cleaned.drop_duplicates(subset=['normalized_title'], keep='first')
        
        # Remove helper columns
        df_deduplicated = df_deduplicated.drop(columns=['normalized_title', 'source_priority', 'has_summary'])
        
        print(f"Before deduplication: {len(df_cleaned):,} records")
        print(f"After deduplication: {len(df_deduplicated):,} records")
        print(f"Removed {len(df_cleaned) - len(df_deduplicated):,} duplicate records")
        
        df_deduplicated = df_deduplicated[~(df_deduplicated.title == "as above")]
        df_deduplicated = df_deduplicated[~(df_deduplicated.title.str.lower().str.contains("equipment grant"))]
        return df_deduplicated

    
    def _clean_nhmrc_grants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean NHMRC grants from nhmrc.org source"""
        if df.empty:
            return df
        
        # Take all NHMRC grants - if source is nhmrc.org, trust it's NHMRC data
        nhmrc_grants = df.copy()
        
        # Standardize columns
        nhmrc_grants.loc[:, 'funding_amount'] = nhmrc_grants.get('nhmrc_funding_amount', pd.NA)
        nhmrc_grants.loc[:, 'for_primary'] = pd.NA  # NHMRC doesn't have FOR codes
        nhmrc_grants.loc[:, 'for'] = pd.NA
        
        nhmrc_grants = nhmrc_grants[~(nhmrc_grants.funder == 'Australian Research Council')] # This will be covered by arc.gov.au source
        
        return nhmrc_grants
    
    def _clean_arc_grants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean ARC grants from arc.gov.au source"""
        if df.empty:
            return df
        
        # Take all ARC grants - if source is arc.gov.au, trust it's ARC data
        arc_grants = df.copy()
        
        # Standardize columns
        arc_grants.loc[:, 'funding_amount'] = arc_grants.get('arc_funding_at_announcement', pd.NA)
        arc_grants.loc[:, 'for_primary'] = arc_grants.get('arc_for_primary', pd.NA)
        arc_grants.loc[:, 'for'] = arc_grants.get('arc_for', pd.NA)
        arc_grants.loc[:, 'funder'] = 'Australian Research Council'
        
        return arc_grants
    
    def _clean_orcid_grants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean ORCID grants - take all from orcid.org source"""
        if df.empty:
            return df
        
        # Take all ORCID grants - if source is orcid.org, trust it's ORCID data
        orcid_grants = df.copy()
        
        # Standardize columns - ORCID grants typically don't have funding amounts or FOR codes
        orcid_grants.loc[:, 'funding_amount'] = pd.NA
        orcid_grants.loc[:, 'for_primary'] = pd.NA
        orcid_grants.loc[:, 'for'] = pd.NA
        
        # Keep original funder name or set to 'ORCID' if missing
        orcid_grants.loc[:, 'funder'] = orcid_grants['funder']
        
        return orcid_grants
    
    def _clean_crossref_grants(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean Crossref grants - take all from crossref.org source"""
        if df.empty:
            return df
        
        # Take all Crossref grants - if source is crossref.org, trust it's Crossref data
        crossref_grants = df.copy()
        
        # Standardize columns - it's fine to keep funding_amount as NA for crossref
        crossref_grants.loc[:, 'funding_amount'] = crossref_grants.get('funding_amount', pd.NA)
        crossref_grants.loc[:, 'for_primary'] = pd.NA
        crossref_grants.loc[:, 'for'] = pd.NA
        
        # Keep original funder name or set to 'Crossref' if missing
        crossref_grants.loc[:, 'funder'] = crossref_grants['funder']
        return crossref_grants

    def _standardize_final_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required columns are present and in the correct order"""
        required_columns = [
            'title', 'grant_summary', 'funding_amount', 'start_year', 'end_year', 
            'funder', 'for_primary', 'for', 'source', 'linked_organizations', 
            'linked_countries', 'linked_researchers'
        ]
        
        # Add missing columns with pd.NA values
        for col in required_columns:
            if col not in df.columns:
                df[col] = pd.NA
        
        # Select final columns in the desired order
        df_final = df[required_columns].copy()
        
        # Fill missing end_year with pd.NA if not present
        if 'end_year' not in df.columns:
            df_final['end_year'] = pd.NA
        
        return df_final


class DataPreprocessor:
    """Main orchestrator for the data preprocessing pipeline"""
    
    def __init__(self, debug: bool = False, force_rerun: bool = False, limit: Optional[int] = None):
        self.debug = debug
        self.force_rerun = force_rerun
        self.limit = limit
        self.output_file = DATA_DIR / "grants_raw.jsonl"
        self.enricher = GrantsEnricher(debug=debug)
        self.cleaner = GrantsCleaner()
        
        # Initialize Neo4j connector
        self.neo4j_connector = Neo4jConnector(
            uri=config.Grants.neo4j_uri,
            username=config.Grants.neo4j_username,
            password=config.Grants.neo4j_password
        )
    
    def save_to_jsonl(self, data: List[dict], filename: Path) -> None:
        """Saves a list of dictionaries to a JSONL file using utils function."""
        if data is not None:
            utils.save_jsonl_file(data, filename)
    
    def load_or_fetch_grants_data(self) -> Optional[List[dict]]:
        """Load existing grants data or fetch from Neo4j"""
        grants_data = None
        
        # Check if grants data already exists and force_rerun is not enabled
        if self.output_file.exists() and not self.force_rerun:
            try:
                grants_data = utils.load_jsonl_file(self.output_file)
                print(f"Loaded existing grants data: {len(grants_data)} records")
            except Exception:
                grants_data = None
        
        # If no existing data, force_rerun is enabled, or loading failed - fetch from Neo4j
        if grants_data is None or self.force_rerun:
            if self.force_rerun:
                print("Force rerun enabled - fetching fresh data from Neo4j...")
            else:
                print("Fetching grants data from Neo4j...")
            
            cipher_query = get_cipher_query(self.limit)
            grants_data = self.neo4j_connector.get_grant_nodes_as_json(cipher_query)
            
            if grants_data is not None:
                self.save_to_jsonl(grants_data, self.output_file)
                print(f"Fetched and saved {len(grants_data)} grants from Neo4j")
            else:
                print("Failed to fetch data from Neo4j database")
                return None
        
        return grants_data
    
    def enrich_grants_data(self, grants_data: List[dict]) -> List[dict]:
        """Enrich grants data with summaries"""
        # Enrich the data with grant summaries from RLA API (skip if enriched file exists and force_rerun is not enabled)
        if config.Grants.enriched_path.exists() and not self.force_rerun:
            print("Loading existing enriched data...")
            enriched_data = utils.load_jsonl_file(config.Grants.enriched_path)
        else:
            if self.force_rerun:
                print("Force rerun enabled - enriching grants data from scratch...")
            else:
                print("Enriching grants data...")
            
            enriched_data = self.enricher.run_async_enrichment(grants_data)
            self.save_to_jsonl(enriched_data, config.Grants.enriched_path)
            print(f"Enriched and saved {len(enriched_data)} grants")
        
        return enriched_data
    
    def load_or_fetch_links_data(self) -> Optional[List[dict]]:
        """Load existing links data or fetch from Neo4j"""
        # Check if links data already exists and force_rerun is not enabled
        if config.Grants.link_path.exists() and not self.force_rerun:
            try:
                links_data = config.Grants.load_links(as_dataframe=False)
                print(f"Loaded existing links data: {len(links_data)} records")
                return links_data
            except Exception:
                pass
        
        # If no existing data, force_rerun is enabled, or loading failed - fetch from Neo4j
        if self.force_rerun:
            print("Force rerun enabled - fetching fresh links data from Neo4j...")
        else:
            print("Fetching links data from Neo4j...")
        
        query = """
        MATCH (o:organisation)--(r:researcher)--(g:grant)
        RETURN o.name as org_name, o.country as org_country, r.orcid as researcher_orcid, g.key as grant_key
        """
        
        try:
            with GraphDatabase.driver(self.neo4j_connector.uri, auth=(self.neo4j_connector.username, self.neo4j_connector.password)) as driver:
                driver.verify_connectivity()
                records, summary, keys = driver.execute_query(query, database_="neo4j")
                
                results = []
                for record in records:
                    result = {
                        "organisation_name": record["org_name"],
                        "organisation_country": record["org_country"],
                        "researcher_orcid": record["researcher_orcid"],
                        "grant_key": record["grant_key"]
                    }
                    results.append(result)
                
                self.save_to_jsonl(results, config.Grants.link_path)
                print(f"Fetched and saved {len(results)} organisation-researcher-grant links")
                return results
                
        except Exception as e:
            print(f"Failed to fetch links data from Neo4j database: {e}")
            return None
    
    def fill_linked_columns(self, cleaned_data: List[dict], links_data: List[dict]) -> List[dict]:
        """Fill linked_* columns in grants data based on grant keys from links data"""
        if not links_data:
            print("No links data available to fill linked columns")
            return cleaned_data
        
        print("Filling linked_* columns based on grant keys...")
        
        # Convert links data to DataFrame for easier processing
        import pandas as pd
        links_df = pd.DataFrame(links_data)
        
        # Group links by grant_key to aggregate linked information
        links_grouped = links_df.groupby('grant_key').agg({
            'organisation_name': lambda x: list(x.dropna().unique()),
            'organisation_country': lambda x: list(x.dropna().unique()),
            'researcher_orcid': lambda x: list(x.dropna().unique())
        }).reset_index()
        
        # Convert to dictionary for faster lookup
        links_dict = {}
        for _, row in links_grouped.iterrows():
            links_dict[row['grant_key']] = {
                'linked_organizations': row['organisation_name'],
                'linked_countries': row['organisation_country'],
                'linked_researchers': row['researcher_orcid']
            }
        
        # Fill linked columns in cleaned data
        updated_data = []
        filled_count = 0
        
        for grant in cleaned_data:
            grant_key = grant.get('id')  # Using 'id' as it's renamed from 'key' in cleaning
            if grant_key and grant_key in links_dict:
                grant['linked_organizations'] = links_dict[grant_key]['linked_organizations']
                grant['linked_countries'] = links_dict[grant_key]['linked_countries']
                grant['linked_researchers'] = links_dict[grant_key]['linked_researchers']
                filled_count += 1
            else:
                # Keep existing values or set to empty lists if not present
                grant['linked_organizations'] = grant.get('linked_organizations', [])
                grant['linked_countries'] = grant.get('linked_countries', [])
                grant['linked_researchers'] = grant.get('linked_researchers', [])
            
            updated_data.append(grant)
        
        print(f"Filled linked columns for {filled_count} out of {len(cleaned_data)} grants")
        return updated_data

    def clean_and_save_grants(self, enriched_data: List[dict], links_data: Optional[List[dict]] = None) -> int:
        """Clean enriched data, fill linked columns, and save final grants file"""
        # Always clean and save when called, but show appropriate message for force_rerun
        if self.force_rerun:
            print("Force rerun enabled - cleaning grants data from scratch...")
        else:
            print("Cleaning grants data...")
        
        df_cleaned = self.cleaner.clean_enriched_data(config.Grants.enriched_path)
        
        if df_cleaned is not None:
            cleaned_data = df_cleaned.reset_index().to_dict('records')
            
            # Fill linked columns if links data is available
            if links_data:
                cleaned_data = self.fill_linked_columns(cleaned_data, links_data)
            
            utils.save_jsonl_file(cleaned_data, config.Grants.grants_path)
            print(f"Cleaned and saved {len(df_cleaned)} grants")
            return len(df_cleaned)
        else:
            print("Grants processing completed with errors (cleaning failed)")
            return 0
    
    def run_pipeline(self) -> None:
        """Run the complete data preprocessing pipeline"""
        print("=== Research Link Technology Landscaping - Data Preprocessing ===")
        if self.force_rerun:
            print("🔄 FORCE RERUN MODE - All data will be regenerated from scratch")
        
        # Process grants data
        grants_data = self.load_or_fetch_grants_data()
        if grants_data is None:
            print("Failed to load or fetch grants data. Exiting.")
            return
        
        # Process links data
        links_data = self.load_or_fetch_links_data()
        if links_data is None:
            print("Warning: Failed to load or fetch links data. Continuing without linked columns.")
        
        enriched_data = self.enrich_grants_data(grants_data)
        cleaned_count = self.clean_and_save_grants(enriched_data, links_data)
        
        print(f"Grants processed: original={len(grants_data)}, enriched={len(enriched_data)}, cleaned={cleaned_count}")
        if links_data:
            print(f"Links processed: {len(links_data)} organisation-researcher-grant relationships")
        
        print("\n" + "="*70)
        print("🎉 Data preprocessing pipeline completed!")
        print("="*70)


app = typer.Typer(help="Research Link Technology Landscaping - Data Preprocessing")


@app.command()
def prepare(
    force_rerun: bool = typer.Option(
        False, 
        "--force-rerun", 
        help="Force rerun all steps including Neo4j query, ignoring existing data files"
    ),
    debug: bool = typer.Option(
        False, 
        "--debug", 
        help="Enable debug output"
    ),
    limit: Optional[int] = typer.Option(
        None, 
        "--limit", 
        help="Limit the number of grants retrieved from Neo4j (default: no limit, retrieves all grants)"
    )
):
    """Run the data preprocessing pipeline"""
    # Create and run the data preprocessing pipeline
    preprocessor = DataPreprocessor(debug=debug, force_rerun=force_rerun, limit=limit)
    preprocessor.run_pipeline()


@app.command()
def link(
    output_file: str = typer.Option(
        config.Grants.link_path,
        "--output",
        "-o",
        help="Output JSONL file name (will be saved in data directory)"
    )
):
    """Export all organisation-researcher-grant links from Neo4j database"""
    query = """
    MATCH (o:organisation)--(r:researcher)--(g:grant)
    RETURN o.name as org_name, o.country as org_country, r.orcid as researcher_orcid, g.key as grant_key
    """
    
    try:
        connector = Neo4jConnector(
            config.Grants.neo4j_uri,
            config.Grants.neo4j_username,
            config.Grants.neo4j_password
        )
        
        with GraphDatabase.driver(connector.uri, auth=(connector.username, connector.password)) as driver:
            driver.verify_connectivity()
            records, summary, keys = driver.execute_query(query, database_="neo4j")
            
            results = []
            for record in records:
                result = {
                    "organisation_name": record["org_name"],
                    "organisation_country": record["org_country"],
                    "researcher_orcid": record["researcher_orcid"],
                    "grant_key": record["grant_key"]
                }
                results.append(result)
            
            output_path = DATA_DIR / output_file
            utils.save_jsonl_file(results, output_path)
            
            print(f"✅ Saved {len(results)} organisation-researcher-grant links to {output_path}")
            print(f"📊 Query executed: Found all relationships between organisations, researchers, and grants")
            
            # Show some summary statistics
            if results:
                countries = set(r["organisation_country"] for r in results if r["organisation_country"])
                orgs = set(r["organisation_name"] for r in results if r["organisation_name"])
                grants = set(r["grant_key"] for r in results if r["grant_key"])
                researchers = set(r["researcher_orcid"] for r in results if r["researcher_orcid"])
                
                print(f"📈 Summary: {len(countries)} countries, {len(orgs)} organisations, {len(researchers)} researchers, {len(grants)} grants")
            
    except Exception as e:
        print(f"❌ Error querying Neo4j database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    app()
    

