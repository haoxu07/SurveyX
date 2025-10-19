import requests
import json
import os
import datetime
import time
from typing import List, Dict, Optional


class DataFetcherForSemantics:
    """
    A class for fetching academic paper data from various APIs.
    Supports Semantic Scholar, Crossref, and OpenAlex APIs.
    """
    
    def __init__(self, output_dir: str = "paper_searches"):
        """
        Initialize the DataFetcherForSemantics.
        
        Args:
            output_dir (str): Directory to save search results
        """
        self.output_dir = output_dir
        self.session = requests.Session()
        
        # Set up headers for API requests
        self.headers = {
            'User-Agent': 'DataFetcherForSemanticsBot/1.0 (mailto:user@example.com)'
        }
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Default retry settings
        self.default_max_retries = 3
        self.default_retry_delay = 2.0
    
    def configure_retry_settings(self, max_retries: int = 3, retry_delay: float = 2.0):
        """
        Configure default retry settings for the fetcher.
        
        Args:
            max_retries (int): Default maximum number of retry attempts
            retry_delay (float): Default base delay for exponential backoff in seconds
        """
        self.default_max_retries = max_retries
        self.default_retry_delay = retry_delay
        print(f"⚙️ Retry settings configured: max_retries={max_retries}, retry_delay={retry_delay}s")
    
    def fetch_abstract_by_doi(self, doi: str) -> Optional[str]:
        """
        Fetch abstract for a paper using its DOI via OpenAlex API.
        
        Args:
            doi (str): The DOI of the paper
        
        Returns:
            str: Abstract text or None if not found
        """
        try:
            # OpenAlex API endpoint for getting work by DOI
            url = f"https://api.openalex.org/works/https://doi.org/{doi}"
            
            print(f"  📖 Fetching abstract for DOI: {doi}")
            
            response = self.session.get(url, headers=self.headers, timeout=20)
            
            if response.status_code != 200:
                print(f"  ⚠️ OpenAlex API failed for {doi}: {response.status_code}")
                return None
            
            data = response.json()
            
            # Extract abstract from inverted index
            inv = data.get("abstract_inverted_index")
            if inv:
                # Reconstruct plaintext from inverted index
                positions = [(i, word) for word, idxs in inv.items() for i in idxs]
                abstract = " ".join(word for _, word in sorted(positions))
                print(f"  ✅ Abstract found ({len(abstract)} chars)")
                return abstract
            
            print(f"  ⚠️ No abstract found for {doi}")
            return None
            
        except Exception as e:
            print(f"  ✗ Error fetching abstract for {doi}: {e}")
            return None
    
    def enhance_papers_with_abstracts(self, papers: List[Dict]) -> List[Dict]:
        """
        Enhance papers that don't have abstracts by fetching them using DOI.
        Only includes papers that successfully get abstracts or already have them.
        Processes all papers that need abstracts.
        
        Args:
            papers (List[Dict]): List of paper dictionaries
        
        Returns:
            List[Dict]: Papers with abstracts (only papers with successful abstract retrieval)
        """
        print(f"\n📖 Enhancing papers with missing abstracts...")
        
        # Find papers without abstracts but with DOIs
        papers_needing_abstracts = []
        for paper in papers:
            if not paper.get('abstract') and paper.get('doi'):
                papers_needing_abstracts.append(paper)
        
        if not papers_needing_abstracts:
            print("✅ All papers already have abstracts")
            return papers
        
        print(f"📊 Found {len(papers_needing_abstracts)} papers needing abstracts")
        
        # Process all papers that need abstracts
        papers_to_enhance = papers_needing_abstracts
        
        enhanced_count = 0
        
        # Filter out papers that cannot get abstracts
        papers_with_abstracts = []
        
        for i, paper in enumerate(papers_to_enhance, 1):
            doi = paper['doi']
            print(f"  [{i}/{len(papers_to_enhance)}] Processing {doi}")
            
            abstract = self.fetch_abstract_by_doi(doi)
            if abstract:
                paper['abstract'] = abstract
                papers_with_abstracts.append(paper)
                enhanced_count += 1
                print(f"    ✅ Abstract fetched successfully")
            else:
                print(f"    ❌ Could not fetch abstract, skipping this paper")
            
            # Rate limiting - be respectful to OpenAlex
            time.sleep(0.5)
        
        # Keep papers that already had abstracts
        papers_already_with_abstracts = [p for p in papers if p.get('abstract') and p['abstract'].strip()]
        
        # Combine papers that already had abstracts with newly enhanced papers
        final_papers = papers_already_with_abstracts + papers_with_abstracts
        
        print(f"✅ Enhanced {enhanced_count} papers with abstracts")
        print(f"📊 Total papers in result: {len(final_papers)} (kept {len(papers_already_with_abstracts)} with existing abstracts + {len(papers_with_abstracts)} with new abstracts)")
        return final_papers
    
    def search_by_keyword(self, keyword: str, limit: int = 50, save_to_file: bool = True, max_retries: int = None, retry_delay: float = None, enhance_abstracts: bool = True) -> Optional[Dict]:
        """
        Search for papers using a keyword via Semantic Scholar API.
        
        Args:
            keyword (str): The keyword to search for
            limit (int): Maximum number of papers to return (default: 50)
            save_to_file (bool): Whether to save results to a JSON file
            max_retries (int): Maximum number of retry attempts (uses default if None)
            retry_delay (float): Base delay for exponential backoff in seconds (uses default if None)
            enhance_abstracts (bool): Whether to fetch missing abstracts using DOI (default: True)
        
        Returns:
            dict: Dictionary containing search results with paper information
                  or None if the search fails after all retry attempts
        """
        # Use default retry settings if none provided
        if max_retries is None:
            max_retries = self.default_max_retries
        if retry_delay is None:
            retry_delay = self.default_retry_delay
        try:
            print(f"🔍 Searching for papers with keyword: '{keyword}'")
            print(f"📊 Requesting top {limit} papers from Semantic Scholar...")
            
            # Semantic Scholar search endpoint
            search_url = "https://api.semanticscholar.org/graph/v1/paper/search"
            
            # Search parameters
            params = {
                'query': keyword,
                'limit': limit,
                'fields': 'title,abstract,externalIds,year,venue'
            }
            
            print(f"🔗 Search URL: {search_url}")
            print(f"📝 Search parameters: {params}")
            
            # Make the search request with retry logic
            response = None
            last_error = None
            
            for attempt in range(max_retries + 1):
                try:
                    if attempt > 0:
                        # Calculate exponential backoff delay
                        backoff_delay = retry_delay * (2 ** (attempt - 1))
                        print(f"🔄 Retry attempt {attempt}/{max_retries}...")
                        print(f"⏳ Waiting {backoff_delay:.1f} seconds (exponential backoff)...")
                        time.sleep(backoff_delay)
                    
                    response = self.session.get(search_url, params=params, headers=self.headers, timeout=30)
                    
                    if response.status_code == 200:
                        print(f"✅ API request successful on attempt {attempt + 1}")
                        break
                    else:
                        last_error = f"Status code: {response.status_code}, Response: {response.text}"
                        print(f"⚠️ Attempt {attempt + 1} failed: {last_error}")
                        
                        if attempt == max_retries:
                            print(f"✗ All {max_retries + 1} attempts failed. Giving up.")
                            print(f"Final error: {last_error}")
                            return None
                        else:
                            next_delay = retry_delay * (2 ** attempt)
                            print(f"⏳ Next retry in {next_delay:.1f} seconds...")
                            
                except requests.exceptions.RequestException as e:
                    last_error = f"Request error: {e}"
                    print(f"⚠️ Attempt {attempt + 1} failed: {last_error}")
                    
                    if attempt == max_retries:
                        print(f"✗ All {max_retries + 1} attempts failed. Giving up.")
                        print(f"Final error: {last_error}")
                        return None
                    else:
                        next_delay = retry_delay * (2 ** attempt)
                        print(f"⏳ Next retry in {next_delay:.1f} seconds...")
            
            # If we get here, response should be successful
            if not response or response.status_code != 200:
                print(f"✗ Unexpected error: Response is not valid after retries")
                return None
            
            # Parse the response
            data = response.json()
            
            # Extract papers from response
            papers = data.get('data', [])
            total_papers = data.get('total', 0)
            
            print(f"✓ Found {len(papers)} papers (total available: {total_papers})")
            
            # Process papers to extract relevant information
            processed_papers = []
            doi_to_title_map = {}
            
            for paper in papers:
                if isinstance(paper, dict):
                    # Extract DOI
                    doi = None
                    if 'externalIds' in paper and paper['externalIds']:
                        doi = paper['externalIds'].get('DOI')
                    
                    # Extract title
                    title = paper.get('title', 'No title')
                    
                    # Extract other information
                    paper_info = {
                        'doi': doi,
                        'title': title,
                        'abstract': paper.get('abstract'),
                        'year': paper.get('year'),
                        'venue': paper.get('venue'),
                        '_id': title,
                        'keyword': keyword
                    }
                    
                    processed_papers.append(paper_info)
                    
                    # Add to DOI to title mapping if DOI exists
                    if doi:
                        doi_to_title_map[doi] = title
            
            # Enhance papers with missing abstracts if requested
            if enhance_abstracts:
                processed_papers = self.enhance_papers_with_abstracts(processed_papers)
            
            # Create output structure
            output_data = {
                'search_info': {
                    'keyword': keyword,
                    'search_timestamp': datetime.datetime.now().isoformat(),
                    'api_source': 'Semantic Scholar',
                    'papers_requested': limit,
                    'papers_returned': len(papers),
                    'total_available': total_papers,
                    'abstracts_enhanced': enhance_abstracts
                },
                'papers': processed_papers,
                'doi_to_title_map': doi_to_title_map,
                'summary': {
                    'papers_with_doi': len(doi_to_title_map),
                    'papers_with_abstract': len([p for p in processed_papers if p.get('abstract')]),
                    'papers_with_venue': len([p for p in processed_papers if p.get('venue')]),
                    'year_range': {
                        'min': min([p.get('year', 9999) for p in processed_papers if p.get('year')], default=None),
                        'max': max([p.get('year', 0) for p in processed_papers if p.get('year')], default=None)
                    }
                }
            }
            
            # Save to file if requested
            if save_to_file:
                # Create safe filename from keyword
                safe_keyword = keyword.replace(' ', '_').replace('/', '_').replace(':', '_').replace('.', '_')
                safe_keyword = ''.join(c for c in safe_keyword if c.isalnum() or c in '_')
                
                json_filename = f"search_{safe_keyword}.json"
                json_path = os.path.join(self.output_dir, json_filename)
                
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                print(f"✓ Search results saved to: {json_path}")
            
            # Print summary
            print(f"\n📊 Search Summary:")
            print(f"   Papers found: {len(processed_papers)}")
            print(f"   Papers with DOI: {len(doi_to_title_map)}")
            print(f"   Papers with abstract: {output_data['summary']['papers_with_abstract']}")
            print(f"   Papers with venue: {output_data['summary']['papers_with_venue']}")
            
            if output_data['summary']['year_range']['min'] and output_data['summary']['year_range']['max']:
                print(f"   Year range: {output_data['summary']['year_range']['min']} - {output_data['summary']['year_range']['max']}")
            
            # Show first few papers
            if processed_papers:
                print(f"\n📄 First 5 papers:")
                for i, paper in enumerate(processed_papers[:5], 1):
                    doi_str = f" (DOI: {paper['doi']})" if paper['doi'] else ""
                    year_str = f" ({paper['year']})" if paper['year'] else ""
                    print(f"   {i}. {paper['title']}{year_str}{doi_str}")
            
            return output_data
            
        except requests.exceptions.RequestException as e:
            print(f"✗ Request error: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"✗ JSON decode error: {e}")
            return None
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return None
    
    def search_multiple_keywords(self, keywords: List[str], limit_per_keyword: int = 50, max_retries: int = None, retry_delay: float = None, enhance_abstracts: bool = True) -> Dict:
        """
        Search for papers using multiple keywords.
        
        Args:
            keywords (List[str]): List of keywords to search for
            limit_per_keyword (int): Maximum number of papers per keyword
            max_retries (int): Maximum number of retry attempts per keyword (uses default if None)
            retry_delay (float): Base delay for exponential backoff in seconds (uses default if None)
            enhance_abstracts (bool): Whether to fetch missing abstracts using DOI (default: True)
        
        Returns:
            dict: Combined results from all keyword searches
        """
        # Use default retry settings if none provided
        if max_retries is None:
            max_retries = self.default_max_retries
        if retry_delay is None:
            retry_delay = self.default_retry_delay
        print(f"🔍 Starting multi-keyword search for {len(keywords)} keywords...")
        
        all_results = {
            'searches': [],
            'all_papers': [],
            'all_doi_to_title_map': {},
            'search_timestamp': datetime.datetime.now().isoformat()
        }
        
        for i, keyword in enumerate(keywords, 1):
            print(f"\n[{i}/{len(keywords)}] Searching for: '{keyword}'")
            
            result = self.search_by_keyword(keyword, limit=limit_per_keyword, save_to_file=True, max_retries=max_retries, retry_delay=retry_delay, enhance_abstracts=enhance_abstracts)
            
            if result:
                all_results['searches'].append({
                    'keyword': keyword,
                    'papers_found': len(result['papers']),
                    'papers_with_doi': len(result['doi_to_title_map'])
                })
                
                # Add papers to combined results
                all_results['all_papers'].extend(result['papers'])
                all_results['all_doi_to_title_map'].update(result['doi_to_title_map'])
            else:
                all_results['searches'].append({
                    'keyword': keyword,
                    'papers_found': 0,
                    'papers_with_doi': 0,
                    'error': 'Search failed'
                })
        
        # Remove duplicate papers based on DOI
        unique_papers = []
        seen_dois = set()
        
        for paper in all_results['all_papers']:
            doi = paper.get('doi')
            if doi and doi not in seen_dois:
                unique_papers.append(paper)
                seen_dois.add(doi)
            elif not doi:
                # Papers without DOI are always included
                unique_papers.append(paper)
        
        all_results['all_papers'] = unique_papers
        
        # Save combined results
        combined_filename = f"combined_search_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        combined_path = os.path.join(self.output_dir, combined_filename)
        
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Multi-keyword search complete!")
        print(f"📊 Total unique papers found: {len(unique_papers)}")
        print(f"📄 Combined results saved to: {combined_path}")
        
        return all_results


def test_data_fetcher():
    """
    Test function to demonstrate the DataFetcherForSemantics functionality.
    """
    print("=" * 60)
    print("Testing DataFetcherForSemantics Class")
    print("=" * 60)
    
    # Initialize the fetcher
    fetcher = DataFetcherForSemantics(output_dir="paper_searches")
    
    # Configure retry settings with exponential backoff (optional)
    print("\n⚙️ Configuring retry settings with exponential backoff...")
    fetcher.configure_retry_settings(max_retries=5, retry_delay=4.0)
    
    # Test multiple keywords search
    print("\n" + "=" * 60)
    print("Example: Multiple keywords search with retry functionality")
    print("=" * 60)
    
    keywords = ["performance montioring unit", "sampling based profiling", "call path profiling", "Precise event sampling"]
    multi_result = fetcher.search_multiple_keywords(keywords, limit_per_keyword=50)
    
    print(f"\n✅ Multi-keyword search completed successfully!")


if __name__ == "__main__":
    test_data_fetcher()
