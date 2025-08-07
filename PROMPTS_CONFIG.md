# Parameterized Prompts Configuration

The keyword extraction and harmonisation workflows now support fully customizable prompts through the `config.yaml` file. This allows you to control the behavior, focus, and output of the LLM tasks without modifying code.

## Keywords Extraction Configuration

Control how many keywords are extracted and what the LLM focuses on:

```yaml
modeling:
  extraction:
    # Number of keywords per category
    keywords_per_category: 8        # Target number (default: 8)
    max_keywords_per_category: 10   # Maximum allowed (default: 10)  
    min_keywords_per_category: 5    # Minimum required (default: 5)
    
    # System message for the LLM
    system_message: |
      Your custom system message defining the LLM's role and expertise...
    
    # What the LLM should focus on when extracting
    focus_areas:
      - "emerging research domains and interdisciplinary areas"
      - "novel methodologies and cutting-edge approaches"
      - "innovative technologies and emerging tools"
      - "novel applications and emerging needs"
    
    # Define the keyword categories to extract
    keyword_categories:
      - name: "keywords"
        description: "most relevant keywords capturing the research content"
      - name: "methodology_keywords" 
        description: "keywords related to research methodologies or approaches"
      - name: "application_keywords"
        description: "keywords related to target applications or outcomes"
      - name: "technology_keywords"
        description: "keywords related to technologies or tools mentioned"
```

## Keywords Harmonisation Configuration

Control how keywords are merged and standardized:

```yaml
modeling:
  harmonisation:
    # Harmonisation parameters (for future similarity-based features)
    max_iterations: 3
    similarity_threshold: 0.8
    merge_threshold: 0.9
    
    # System message for the LLM
    system_message: |
      Your custom system message for the harmonisation expert...
    
    # Instructions for the harmonisation process
    instructions:
      - "Identify and merge similar keywords, variants, and synonyms"
      - "Standardize terminology using commonly accepted scientific terms"
      - "Resolve inconsistencies in naming conventions"
      - "Preserve semantic richness while reducing redundancy"
      - "Use the most standard, widely-accepted term when merging"
      - "Preserve technical precision - don't over-generalize"
      - "Maintain appropriate granularity level for research analysis"
```

## Usage Examples

### 1. Use default configuration:
```bash
python modeling/cli.py extract
python modeling/cli.py harmonise
```

### 2. Use custom configuration:
```bash
python modeling/cli.py extract --config my-config.yaml
python modeling/cli.py harmonise --config my-config.yaml
```

### 3. Override just the model:
```bash
python modeling/cli.py extract --model openai/gpt-4o
```

## Configuration Tips

**For Keywords Extraction:**
- Increase `keywords_per_category` for more comprehensive extraction
- Customize `focus_areas` to target specific research domains
- Modify `system_message` to emphasize particular expertise areas
- Add/remove `keyword_categories` based on your analysis needs

**For Keywords Harmonisation:**
- Adjust `instructions` to be more/less aggressive in merging
- Customize `system_message` for domain-specific expertise
- Use different parameters for different research fields

**Example: Focus on Medical Research**
```yaml
modeling:
  extraction:
    keywords_per_category: 10
    focus_areas:
      - "clinical applications and medical technologies"
      - "biomedical methodologies and approaches"  
      - "healthcare innovations and therapeutic tools"
    system_message: |
      You are a medical research analyst with expertise in clinical studies,
      biomedical technologies, and healthcare innovations...
```

**Example: Conservative Harmonisation**
```yaml  
modeling:
  harmonisation:
    system_message: |
      Be very conservative in keyword merging. Only merge terms that are
      clearly identical concepts with different spellings or abbreviations.
    instructions:
      - "Merge only obvious variants and abbreviations"
      - "Preserve all technical distinctions"
      - "When in doubt, keep keywords separate"
```

The parameterized approach gives you full control over the LLM behavior while maintaining the structured workflow and output formats.
