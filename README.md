# Deep Research Agent

An AI-powered autonomous research assistant that performs deep, multi-source academic research using web search (Tavily), ArXiv papers, and PubMed articles. Built with LangGraph and designed for comprehensive, validated research with detailed source attribution and academic focus.

## 🌟 Features

- **Multi-Source Research**: Seamlessly integrates web search (Tavily), ArXiv papers, and PubMed articles
- **Deep Research**: Configurable research depth (1-5 levels) with iterative refinement
- **Academic Validation**: Specialized validation criteria for academic and biomedical content
- **Detailed Source Attribution**: Complete source tracking with detailed metadata
- **Smart Summarization**: AI-powered content summarization optimized for academic papers
- **Rate Limit Resilience**: Advanced retry mechanisms with exponential backoff
- **Structured Output**: JSON-formatted results with comprehensive source information
- **Academic Focus**: Specialized handling for academic papers, preprints, and medical literature
- **Comprehensive Testing**: Built-in testing suite for all components

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd deep-research-agent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create .env file
echo "TAVILY_API_KEY=your_tavily_key_here" > .env
echo "GROQ_API_KEY=your_groq_key_here" >> .env
```

### Required Dependencies

```python
# Core dependencies
langgraph>=0.1.0
langchain-groq>=0.1.0
langchain-community>=0.1.0
tavily-python>=0.3.0
arxiv>=2.1.0
requests>=2.31.0
beautifulsoup4>=4.12.0
scholarly>=1.7.0
python-dotenv>=1.0.0
```

### Running the Agent

```python
from deep_agent1 import research_graph

# Basic usage
result = research_graph.invoke({
    "question": "How is CRISPR being used in current medical treatments?",
    "max_depth": 2,
    "max_results": 5
})

print(f"Research completed with score: {result['validation_score']:.2f}")
print(f"Sources used: {result['sources_used']}")
print(f"Total sources: {result['sources_used_count']}")
print(f"Answer: {result['refined_answer']}")
```

### Running Comprehensive Tests

```python
from deep_agent1 import comprehensive_test

# Run full system test
comprehensive_test()
```

## 🏗️ Architecture Overview

### Enhanced Research Pipeline

The agent follows a sophisticated 7-node graph-based workflow:

1. **Research Node**: Unified search across web, ArXiv, and PubMed with detailed metadata
2. **Draft Node**: Creates initial answer optimized for academic content
3. **Generate Follow-ups Node**: Identifies research gaps and generates targeted questions
4. **Research Follow-ups Node**: Performs additional research on identified gaps
5. **Refine Node**: Refines answer with proper citations and source attribution
6. **Validation Node**: Specialized academic validation with scoring
7. **Improve Research Node**: Generates improvement questions based on validation feedback

### Enhanced ResearchState

```python
class ResearchState(TypedDict):
    question: str                    # Original research question
    search_results: List[str]        # Summarized search results
    drafted_content: str              # Initial draft answer
    refined_answer: str              # Final refined answer
    follow_up_questions: List[str]   # Generated follow-up questions
    research_depth: int              # Current research depth
    max_depth: int                  # Maximum research depth (1-5)
    validation_score: float          # Quality score (0.0-1.0)
    validation_feedback: str          # Detailed validation feedback
    needs_more_research: bool        # Flag for additional research
    sources_used: List[str]          # List of source types used
    source_counts: dict              # Count by source type
    detailed_sources: List[dict]     # Complete source metadata
    max_results: int                # Results per source type
```

### Multi-Source Integration

#### Web Search (Tavily)
- General web content and news articles
- Real-time information updates
- Comprehensive topic coverage

#### ArXiv Integration
- Latest research papers and preprints
- Physics, mathematics, computer science focus
- Paper metadata: title, authors, abstract, publication date, DOI

#### PubMed Integration
- Biomedical and medical literature
- Author information, abstracts, publication dates
- DOI and PubMed ID tracking

### LLM Configuration

- **Primary LLM**: ChatGroq with "openai/gpt-oss-120b" for research and drafting
- **Validation LLM**: ChatGroq with "moonshotai/kimi-k2-instruct" for quality assessment
- **Temperature**: 0 for consistent, factual responses
- **Retry Logic**: Exponential backoff with 3 retries

## 📊 Enhanced Validation Criteria

### Academic-Specific Validation
Research is validated across dimensions tailored for academic content:

1. **Accuracy of Scientific Concepts**: Correct interpretation of technical terms
2. **Research Findings Interpretation**: Proper understanding of study results
3. **Limitation Recognition**: Acknowledgment of study constraints
4. **Citation Quality**: Proper academic referencing
5. **Explanation Clarity**: Clear communication of complex concepts

### Validation Score Interpretation
- **0.8-1.0**: Excellent quality research
- **0.6-0.79**: Good quality, minor improvements possible
- **0.4-0.59**: Adequate, may benefit from additional research
- **<0.4**: Requires significant improvement

## 📤 Enhanced Output Format

### Structured JSON Response

```json
{
  "question": "Original research question",
  "refined_answer": "Comprehensive research answer with citations",
  "validation_score": 0.92,
  "validation_feedback": "Detailed feedback on quality assessment",
  "research_depth": 2,
  "sources_used": ["Web", "arXiv", "PubMed"],
  "sources_used_count": 15,
  "source_counts": {
    "Web": 8,
    "arXiv": 4,
    "PubMed": 3
  },
  "detailed_sources": [
    {
      "source_type": "arXiv",
      "title": "Paper Title",
      "authors": ["Author1", "Author2"],
      "published": "2024-01-15",
      "url": "https://arxiv.org/abs/...",
      "doi": "arXiv:2401.12345",
      "content": "Paper abstract preview..."
    }
  ]
}
```

### Source Attribution Features
- **Detailed source descriptions** for each type
- **Author information** for academic papers
- **Publication dates** for recency assessment
- **DOI/URL tracking** for verification
- **Content previews** for quick reference

## 🔧 Advanced Configuration

### Custom Research Parameters

```python
# Configure research depth and source limits
result = research_graph.invoke({
    "question": "Your research question",
    "max_depth": 3,           # 1-5 levels of research depth
    "max_results": 5          # Results per source type
})
```

### Source Prioritization

```python
# Example showing source usage distribution
config = {
    "question": "Quantum computing applications",
    "max_depth": 2,
    "max_results": 3
}
# System automatically balances sources based on topic relevance
```

### Error Handling and Reliability

- **Safe API calls** with automatic retry
- **Graceful degradation** when sources are unavailable
- **Comprehensive error logging** for debugging
- **Fallback strategies** for failed searches

## 🧪 Comprehensive Testing

### Test Suite Components

#### Individual Source Testing
```python
# Test each source independently
arxiv_results = search_arxiv("neural networks", max_results=3)
pubmed_results = search_pubmed("machine learning", max_results=3)
unified_results = unified_search("AI applications", max_results=5)
```

#### Full Pipeline Testing
```python
# Complete system test with academic focus
test_question = "Explain how transformer architectures are used in drug discovery"
result = research_graph.invoke({
    "question": test_question,
    "max_depth": 2,
    "max_results": 3
})
```

#### Performance Metrics
- **Time per research phase** tracking
- **API call counts** by source type
- **Success rates** for each search provider
- **Validation score distribution** analysis

## 📈 Example Use Cases

### Academic Literature Review
```python
# Comprehensive review of recent advances
result = research_graph.invoke({
    "question": "Recent advances in transformer architectures for protein folding prediction",
    "max_depth": 3,
    "max_results": 5
})
```

### Medical Research Analysis
```python
# Biomedical research with PubMed focus
result = research_graph.invoke({
    "question": "CRISPR applications in treating genetic disorders: current status and challenges",
    "max_depth": 2,
    "max_results": 4
})
```

### Technology Assessment
```python
# Comparative analysis with academic sources
result = research_graph.invoke({
    "question": "Compare GPT-based models with traditional NLP approaches for scientific text analysis",
    "max_depth": 2,
    "max_results": 3
})
```

## 🔮 Future Enhancements

### Planned Improvements
- **Semantic search** integration for better relevance
- **PDF processing** for direct academic paper analysis
- **Citation network** building for research paper relationships
- **Real-time monitoring** for research topic updates
- **Multi-language support** for non-English academic sources
- **Custom source plugins** for specialized databases

### Scalability Features
- **Asynchronous processing** for large-scale research
- **Caching mechanisms** for frequently accessed sources
- **Rate limit optimization** across all APIs
- **Parallel processing** for multiple research queries

## 🤝 Contributing

Contributions are welcome! Areas for contribution:
- Additional academic source integrations
- Enhanced validation algorithms
- Performance optimizations
- Testing improvements
- Documentation enhancements

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangGraph Team**: For the excellent graph-based AI framework
- **Tavily**: For reliable web search API
- **ArXiv**: For open access to cutting-edge research papers
- **PubMed**: For comprehensive biomedical literature access
- **Groq**: For high-performance LLM inference
- **Academic Community**: For the open science movement making this research possible