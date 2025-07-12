# Influencer Discovery Tool

An AI-powered influencer discovery application that enables semantic search across influencer data using multimodal embeddings. Built with FastAPI, CLIP for text and image understanding, and FAISS for efficient vector search.

## Features

- **Multimodal Semantic Search**: Find influencers using natural language queries that search across text (bios, descriptions) and images (profile photos, content thumbnails)
- **Real-time Web Interface**: Modern, responsive web UI with search suggestions and filtering options
- **Advanced Filtering**: Filter by category, follower count ranges, and other metadata
- **Fast Vector Search**: Efficient FAISS-based similarity search with configurable weights for text, profile, and content embeddings
- **RESTful API**: Complete API for programmatic access to search functionality
- **Automatic Data Processing**: Built-in Instagram data processor with CSV support
- **Health Monitoring**: System health checks and statistics endpoints

## Technology Stack

- **Backend**: FastAPI with Python 3.8+
- **AI/ML**: CLIP (Contrastive Language-Image Pre-training) for multimodal embeddings
- **Vector Search**: FAISS for efficient similarity search
- **Frontend**: Bootstrap 5 with vanilla JavaScript
- **Data Processing**: Pandas for CSV handling and data transformation
- **Storage**: Local file-based vector index with metadata persistence

## Getting Started

### Prerequisites

- Python 3.8 or higher
- UV package manager (recommended) or pip

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd influencer_search_app
   ```

2. **Install dependencies**:
   ```bash
   # Using UV (recommended)
   uv sync
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Configure environment** (optional):
   Create a `.env` file in the root directory:
   ```env
   # CLIP Configuration
   CLIP_MODEL=openai/clip-vit-base-patch32
   CLIP_DEVICE=cpu  # or "cuda" for GPU acceleration
   
   # Search Configuration
   SIMILARITY_THRESHOLD=0.7
   MAX_SEARCH_RESULTS=50
   DEFAULT_SEARCH_LIMIT=10
   
   # Embedding Weights (must sum to 1.0)
   TEXT_WEIGHT=0.4
   PROFILE_WEIGHT=0.3
   CONTENT_WEIGHT=0.3
   
   # Performance
   BATCH_SIZE=32
   CACHE_EMBEDDINGS=true
   ```

### Data Setup

The application comes with sample Instagram influencer data. The system will automatically:

1. **Load existing data** if available
2. **Process the CSV file** (`data/instagram_influencers_final_20250708.csv`) on first run
3. **Generate embeddings** for text, profile images, and content images
4. **Create a vector index** for fast similarity search

### Running the Application

1. **Development mode**:
   ```bash
   uv run fastapi dev
   ```

2. **Production mode**:
   ```bash
   uv run fastapi run
   ```

3. **Direct Python execution**:
   ```bash
   python main.py
   ```

The application will be available at [http://localhost:8000](http://localhost:8000)

## Usage

### Web Interface

1. **Search Interface**: Enter natural language queries like:
   - "fitness influencer with curly hair"
   - "beauty creator who posts makeup tutorials"
   - "lifestyle blogger in New York"

2. **Filters**: Use the sidebar filters to narrow results by:
   - Category (Fitness, Beauty, Lifestyle, Food, Tech, Gaming)
   - Follower count range
   - Verification status

3. **Results**: View influencer cards with:
   - Profile photo and content thumbnail
   - Bio and engagement metrics
   - Similarity score and match reasons
   - Direct Instagram links

### API Endpoints

#### Search Influencers
```bash
POST /api/search
Content-Type: application/json

{
  "query": "fitness influencer with curly hair",
  "limit": 10,
  "category": "Fitness",
  "min_followers": 10000,
  "max_followers": 1000000
}
```

#### Get Categories
```bash
GET /api/categories
```

#### Get Search Suggestions
```bash
GET /api/suggestions?q=fitness
```

#### System Health Check
```bash
GET /api/health
```

#### System Statistics
```bash
GET /api/stats
```

### Example API Usage

```python
import requests

# Search for fitness influencers
response = requests.post('http://localhost:8000/api/search', json={
    'query': 'fitness influencer with curly hair',
    'limit': 5,
    'category': 'Fitness'
})

results = response.json()
for influencer in results['results']:
    print(f"{influencer['name']} - {influencer['follower_count']} followers")
```

## Data Structure

The application works with Instagram influencer data containing:

### Required Fields
- **Handle**: Instagram username
- **Full_Name**: Display name
- **Category**: Content category (Fitness, Beauty, Lifestyle, etc.)
- **Bio**: Instagram bio text
- **Follower_Count**: Number of followers
- **Profile_Photo_URL**: Profile image URL
- **Latest_Post_Thumbnail**: Content thumbnail URL

### Optional Fields
- **Instagram_URL**: Direct profile link
- **Following_Count**: Number of accounts followed
- **Post_Count**: Number of posts
- **Is_Verified**: Verification status
- **Is_Private**: Private account status

## Architecture

### Core Components

1. **Search Engine** (`app/search_engine.py`): High-level search orchestration
2. **Vector Store** (`app/vector_store.py`): FAISS-based similarity search
3. **Embedding Service** (`app/embedding_service.py`): CLIP model integration
4. **Data Processor** (`app/instagram_data_processor.py`): CSV data processing
5. **Web Interface** (`templates/index.html`): Bootstrap-based UI

### Search Process

1. **Query Processing**: Convert natural language to CLIP text embedding
2. **Multimodal Search**: Compare against text, profile, and content embeddings
3. **Weighted Scoring**: Combine similarity scores using configurable weights
4. **Filtering**: Apply metadata filters (category, followers, etc.)
5. **Ranking**: Sort results by relevance score
6. **Response**: Return formatted results with match explanations

## Configuration

### Embedding Weights
Configure the importance of different data types in search:
- `TEXT_WEIGHT`: Bio and description similarity (default: 0.4)
- `PROFILE_WEIGHT`: Profile photo similarity (default: 0.3)
- `CONTENT_WEIGHT`: Content thumbnail similarity (default: 0.3)

### Performance Settings
- `BATCH_SIZE`: Embedding generation batch size
- `CACHE_EMBEDDINGS`: Enable/disable embedding caching
- `MAX_WORKERS`: Number of parallel processing workers

## Development

### Project Structure
```
influencer_search_app/
├── app/                    # Core application modules
│   ├── search_engine.py   # Search orchestration
│   ├── vector_store.py    # FAISS vector operations
│   ├── embedding_service.py # CLIP model integration
│   ├── data_processor.py  # Data processing utilities
│   └── settings.py        # Configuration management
├── data/                  # Sample data and images
├── templates/             # HTML templates
├── static/               # CSS, JS, and static assets
├── main.py               # FastAPI application entry point
└── pyproject.toml        # Project dependencies
```

### Testing
```bash
# Run comprehensive tests
python test_comprehensive_system.py

# Run multimodal search tests
python test_multimodal_search.py
```

### Adding New Data Sources

1. **CSV Format**: Follow the Instagram data structure
2. **Image URLs**: Ensure profile and content images are accessible
3. **Data Processing**: Use the existing processor or create custom handlers

## Performance

- **Search Speed**: Sub-second response times for typical queries
- **Scalability**: FAISS index supports millions of influencers
- **Memory Usage**: Efficient embedding storage and retrieval
- **GPU Support**: Optional CUDA acceleration for CLIP model

## Troubleshooting

### Common Issues

1. **No search results**: Check if data is properly loaded via `/api/health`
2. **Slow performance**: Consider using GPU acceleration (`CLIP_DEVICE=cuda`)
3. **Memory issues**: Reduce `BATCH_SIZE` or `MAX_SEARCH_RESULTS`
4. **Image loading errors**: Verify image URLs are accessible

### Logs
Application logs provide detailed information about:
- Data loading status
- Search query processing
- Embedding generation
- Performance metrics

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **CLIP Model**: OpenAI's Contrastive Language-Image Pre-training
- **FAISS**: Facebook AI Similarity Search for vector operations
- **FastAPI**: Modern web framework for building APIs
- **Bootstrap**: Frontend framework for responsive design
