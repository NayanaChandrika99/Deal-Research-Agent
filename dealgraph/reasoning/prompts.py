# ABOUTME: Prompt management system for deal reasoning with versioned templates and metadata.
# ABOUTME: Handles loading, versioning, and management of reasoning prompts.

from typing import Optional, Dict, Any, List
from pathlib import Path
import json
import re
from dataclasses import dataclass, asdict


@dataclass
class PromptMetadata:
    """Metadata for a prompt version."""
    version: str
    name: str
    description: str
    model: str
    temperature: float = 0.7
    max_tokens: int = 2000
    created_date: str = ""
    performance_metrics: Optional[Dict[str, Any]] = None
    changelog_entry: Optional[str] = None
    system_prompt: Optional[str] = None


class PromptRegistry:
    """
    Manages prompt versions and provides loading functionality.
    
    Supports semantic versioning and metadata tracking for prompt optimization.
    """
    
    def __init__(self, prompts_dir: Path):
        """
        Initialize prompt registry.
        
        Args:
            prompts_dir: Directory containing prompt files and metadata
        """
        self.prompts_dir = Path(prompts_dir)
        self._cache = {}  # Cache loaded prompts
        
    def load_prompt(
        self,
        prompt_name: str,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load a prompt by name and version.
        
        Args:
            prompt_name: Name of the prompt (e.g., "deal_reasoner")
            version: Specific version to load (e.g., "v1", "v2"). If None, loads latest.
            
        Returns:
            Dictionary with:
            - content: The prompt template
            - metadata: PromptMetadata object
            - version: Version string
            - file_path: Path to the prompt file
        """
        prompt_dir = self.prompts_dir / prompt_name
        
        if not prompt_dir.exists():
            raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")
        
        # Determine version to load
        if version is None:
            version = self._get_latest_version(prompt_name)
            if version is None:
                raise ValueError(f"No versions found for prompt: {prompt_name}")
        
        # Check cache first
        cache_key = f"{prompt_name}:{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Load prompt file
        prompt_file = prompt_dir / f"{version}_naive.txt"
        if not prompt_file.exists():
            # Try optimized version
            prompt_file = prompt_dir / f"{version}_optimized.json"
            if not prompt_file.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        
        # Load content
        if prompt_file.suffix == ".json":
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)
                content = prompt_data.get("content", "")
                metadata_dict = prompt_data.get("metadata", {})
        else:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                metadata_dict = {}
        
        # Load or create metadata
        metadata = self._load_metadata(prompt_name, version, metadata_dict)
        
        # Cache the result
        result = {
            "content": content,
            "metadata": metadata,
            "version": version,
            "file_path": str(prompt_file)
        }
        self._cache[cache_key] = result
        
        return result
    
    def _get_latest_version(self, prompt_name: str) -> Optional[str]:
        """Get the latest version for a prompt."""
        prompt_dir = self.prompts_dir / prompt_name
        
        if not prompt_dir.exists():
            return None
        
        # Parse version numbers from filenames
        versions = []
        for file_path in prompt_dir.glob("*.txt"):
            filename = file_path.name
            if filename.startswith("v") and filename.endswith("_naive.txt"):
                version_num = filename.split("_")[0]
                if self._is_valid_version(version_num):
                    versions.append(version_num)
    
        if not versions:
            # Try optimized files
            for file_path in prompt_dir.glob("*.json"):
                filename = file_path.name
                if filename.startswith("v") and filename.endswith("_optimized.json"):
                    version_num = filename.split("_")[0]
                    if self._is_valid_version(version_num):
                        versions.append(version_num)
        
        if not versions:
            return None
        
        # Return latest version (semantic version comparison)
        return self._get_latest_semantic_version(versions)
    
    def _is_valid_version(self, version: str) -> bool:
        """Check if version string is valid (e.g., v1, v2.0, v1.2.3)."""
        pattern = r'^v\d+(\.\d+)*$'
        return bool(re.match(pattern, version))
    
    def _get_latest_semantic_version(self, versions: List[str]) -> str:
        """Get latest version from semantic version list."""
        # Simple version comparison for now
        # In production, this could use semantic_version library
        def version_key(v):
            # Convert "v1.2.3" to [1, 2, 3]
            nums = [int(x) for x in v[1:].split('.')]
            return nums
        
        return max(versions, key=version_key)
    
    def _load_metadata(
        self,
        prompt_name: str,
        version: str,
        extra_metadata: Dict[str, Any]
    ) -> PromptMetadata:
        """Load metadata for a prompt version."""
        # Default metadata
        metadata = PromptMetadata(
            version=version,
            name=prompt_name,
            description=f"{prompt_name} prompt version {version}",
            model="llama3.1-8b",
            temperature=0.7,
            max_tokens=2000
        )
        
        # Update with loaded metadata
        for key, value in extra_metadata.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)
        
        return metadata
    
    def list_versions(self, prompt_name: str) -> List[str]:
        """List all available versions for a prompt."""
        prompt_dir = self.prompts_dir / prompt_name
        
        if not prompt_dir.exists():
            return []
        
        versions = []
        for file_path in prompt_dir.glob("*.txt"):
            filename = file_path.name
            if filename.startswith("v") and "_naive" in filename:
                version = filename.split("_")[0]
                if self._is_valid_version(version):
                    versions.append(version)
        
        for file_path in prompt_dir.glob("*.json"):
            filename = file_path.name
            if filename.startswith("v") and "_optimized" in filename:
                version = filename.split("_")[0]
                if self._is_valid_version(version):
                    versions.append(version)
        
        return sorted(versions, key=lambda v: self._version_to_tuple(v))
    
    def _version_to_tuple(self, version: str) -> tuple:
        """Convert version string to tuple for sorting."""
        stripped = version[1:] if version.startswith("v") else version
        return tuple(int(x) for x in stripped.split('.'))
    
    def get_metadata(self, prompt_name: str, version: str) -> Optional[PromptMetadata]:
        """Get metadata for a specific prompt version."""
        try:
            prompt_data = self.load_prompt(prompt_name, version)
            return prompt_data["metadata"]
        except:
            return None
    
    def clear_cache(self) -> None:
        """Clear the prompt cache."""
        self._cache.clear()


def load_prompt(
    prompt_name: str,
    version: Optional[str] = None,
    prompts_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Convenience function to load a prompt.
    
    Args:
        prompt_name: Name of the prompt
        version: Version to load (None for latest)
        prompts_dir: Prompts directory (uses default if None)
        
    Returns:
        Prompt data dictionary
    """
    if prompts_dir is None:
        # Use default prompts directory relative to project root
        project_root = Path(__file__).parent.parent.parent
        prompts_dir = project_root / "prompts"
    
    registry = PromptRegistry(prompts_dir)
    return registry.load_prompt(prompt_name, version)


# Default prompt templates
DEAL_REASONER_NAIVE_PROMPT = """
You are a private-equity deal research assistant. Given a new opportunity description and a set of historical deals, your task is to:

1. Identify which historical deals are the best precedents.
2. Explain why they are precedents.
3. Extract common 'playbook levers' used in these deals.
4. Extract common risk themes.
5. Return a JSON object with fields: precedents, playbook_levers, risk_themes, narrative_summary.

New opportunity:
{query}

Candidate deals:
{deals_block}

Each deal has the format:
- id: <deal_id>
- name: <deal_name>
- description: <one-line description>
- snippets: <key paragraphs from news/case studies>
- sector: <sector>
- region: <region>
- status: <current/realized>
- metadata: <buy-and-build / add-ons / exits information>

Please analyze and respond ONLY with valid JSON in the following format:
{{
  "precedents": [
    {{
      "deal_id": "<deal_id>",
      "name": "<deal_name>",
      "similarity_reason": "<explanation of why this is a precedent>"
    }}
  ],
  "playbook_levers": [
    "<common value-creation strategy 1>",
    "<common value-creation strategy 2>"
  ],
  "risk_themes": [
    "<common risk pattern 1>",
    "<common risk pattern 2>"
  ],
  "narrative_summary": "<executive summary synthesizing the analysis>"
}}

Focus on identifying patterns and extracting actionable insights that would be relevant for evaluating the new opportunity.
"""

REVERSE_QUERY_NAIVE_PROMPT = """
You are a reverse-query generator for private-equity deal clusters. Given a cluster of related deals (platform + add-ons), generate realistic user queries that would have led to selecting these deals.

Deal cluster:
{deal_cluster}

Generate 3-5 realistic user queries that a PE professional might ask when looking for precedents similar to this cluster. Each query should:
- Be specific enough to return these deals but general enough to be realistic
- Use industry terminology
- Focus on strategic patterns (sectors, strategies, exit types)
- Be 1-2 sentences long

Respond ONLY with a JSON array of strings:

["query 1", "query 2", "query 3"]

Example format:
["US industrial distribution roll-up with strategic add-ons and private equity exit", "Healthcare platform acquisition strategy in mid-market", "Technology consolidation play with operational improvements"]
"""
