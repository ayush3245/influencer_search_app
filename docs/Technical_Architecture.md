# Technical Architecture: Influencer Discovery System

## System Overview

The Influencer Discovery System is designed to enable semantic search across influencer data using multimodal embeddings. The architecture leverages CLIP for understanding both text and image content, with LlamaIndex providing the search infrastructure.

## Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │    │   Data Sources  │    │  Configuration  │
│                 │    │                 │    │                 │
│ "Find fitness   │    │ • PostgreSQL   │    │ • .env vars     │
│  influencers    │    │ • CSV/Excel    │    │ • Model settings│
│  with curly     │    │ • Image URLs   │    │ • Search params │
│  hair"          │    │ • Metadata     │    │                 │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Application Layer                    │
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Search API   │  │ Upload API   │  │ Config API   │        │
│  │ /search      │  │ /upload      │  │ /health      │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   LlamaIndex Workflow Layer                     │
│                                                                │
│  ┌─────────────────┐    ┌─────────────────┐                   │
│  │ Query Processing│    │ Result Ranking  │                   │
│  │ • Parse query   │    │ • Score results │                   │
│  │ • Extract filters│   │ • Apply filters │                   │
│  │ • Route to search│   │ • Format output │                   │
│  └─────────────────┘    └─────────────────┘                   │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CLIP Embedding Layer                         │
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Text Encoder │  │ Image Encoder│  │ Similarity   │        │
│  │ • Tokenize   │  │ • Preprocess │  │ • Cosine     │        │
│  │ • Embed      │  │ • Embed      │  │ • Ranking    │        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Vector Storage Layer                        │
│                                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Vector Store │  │ Metadata DB  │  │ Cache Layer  │        │
│  │ • Embeddings │  │ • Structured │  │ • Redis      │        │
│  │ • Fast search│  │ • Filters    │  │ • Query cache│        │
│  └──────────────┘  └──────────────┘  └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Data Ingestion Pipeline

**Purpose**: Process and index influencer data from various sources.

**Components**:
- **Data Loaders**: CSV, Excel, PostgreSQL connectors
- **Image Processors**: URL validation, format conversion
- **Text Processors**: Bio/description cleaning and normalization
- **Embedding Generators**: CLIP-based multimodal embeddings

**Flow**:
```python
Raw Data → Validation → Preprocessing → Embedding Generation → Vector Storage
```

**Implementation**:
```python
# In app/ingestion.py
class InfluencerDataPipeline:
    def __init__(self):
        self.clip_model = ClipEmbedding()
        self.vector_store = VectorStoreIndex()
    
    def process_batch(self, data_batch):
        # Process text and images
        # Generate embeddings
        # Store in vector database
```

### 2. Search Engine Architecture

**Query Processing Flow**:
1. **Input Parsing**: Extract search terms and filters
2. **Query Embedding**: Convert search query to CLIP embedding
3. **Vector Search**: Find similar embeddings in vector store
4. **Metadata Filtering**: Apply structured filters (follower count, etc.)
5. **Result Ranking**: Score and sort final results
6. **Response Formatting**: Structure output for API/UI

**Search Types Supported**:
- **Text-to-Text**: "Find beauty influencers" → Match bio descriptions
- **Text-to-Image**: "Find people with curly hair" → Match profile photos
- **Image-to-Image**: Upload reference image → Find similar profiles
- **Hybrid**: "Fitness influencer with 50K+ followers" → Combined search

### 3. Vector Storage Strategy

**Primary Storage**: LlamaIndex VectorStoreIndex with FAISS backend
- Embeddings stored as 512-dimensional vectors (CLIP ViT-B/32)
- Supports similarity search with cosine distance
- Optimized for read-heavy workloads

**Metadata Storage**: Structured data in PostgreSQL/SQLite
- Influencer profiles (name, username, platform)
- Metrics (follower count, engagement rate)
- Demographics (age, location, category)
- Content metadata (post count, recent activity)

**Caching Strategy**:
- **Query Cache**: Redis for frequent search results
- **Embedding Cache**: Pre-computed embeddings for all data
- **Image Cache**: Processed/resized images for faster loading

### 4. CLIP Integration Details

**Model Configuration**:
```python
# In app/settings.py
from llama_index.embeddings.clip import ClipEmbedding

class CLIPConfig:
    model_name = "ViT-B/32"  # or "ViT-L/14" for higher accuracy
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 32
    max_image_size = (224, 224)
```

**Text Processing**:
- Tokenization with CLIP's text encoder
- Max token length: 77 tokens
- Automatic truncation for longer texts
- Template-based prompts for better matching

**Image Processing**:
- Automatic resizing to 224x224 pixels
- Normalization using CLIP's preprocessing
- Support for multiple formats (JPEG, PNG, WebP)
- Error handling for broken/missing images

### 5. Performance Optimization

**Embedding Generation**:
- Batch processing for multiple items
- GPU acceleration when available
- Async processing for I/O-bound operations
- Progress tracking for large datasets

**Search Optimization**:
- Pre-computed embeddings (no real-time encoding)
- Approximate nearest neighbor search (FAISS)
- Query result caching
- Pagination for large result sets

**Resource Management**:
- Model loading on startup (not per request)
- Memory-mapped vector storage
- Connection pooling for databases
- Rate limiting for API endpoints

## Data Flow Diagrams

### Ingestion Flow
```
CSV/Excel/DB → Data Validation → Text/Image Processing → CLIP Embedding → Vector Store
                      ↓
               Metadata Extraction → Structured Storage (PostgreSQL)
```

### Search Flow
```
User Query → Query Parsing → CLIP Embedding → Vector Search → Metadata Filter → Ranking → Results
```

### Real-time Pipeline
```
New Influencer Data → Background Processing → Index Update → Cache Invalidation
```

## Scalability Considerations

### Horizontal Scaling
- **API Layer**: Multiple FastAPI instances behind load balancer
- **Processing**: Distributed embedding generation with Celery
- **Storage**: Sharded vector databases for large datasets
- **Caching**: Redis cluster for distributed caching

### Vertical Scaling
- **GPU Utilization**: CLIP inference on dedicated GPU instances
- **Memory Optimization**: Efficient vector storage formats
- **CPU Optimization**: Multi-threaded similarity search

### Performance Targets
- **Search Latency**: < 200ms for typical queries
- **Throughput**: 100+ concurrent search requests
- **Index Size**: Support for 1M+ influencer profiles
- **Update Frequency**: Real-time ingestion of new data

## Security & Privacy

### Data Protection
- **API Authentication**: JWT-based access control
- **Image Security**: URL validation and content filtering
- **Data Encryption**: At-rest and in-transit encryption
- **Access Logging**: Comprehensive audit trails

### Privacy Compliance
- **Data Minimization**: Only store necessary information
- **Consent Management**: Track data usage permissions
- **Right to Deletion**: Support for data removal requests
- **Anonymization**: Option to hash sensitive identifiers

## Monitoring & Observability

### Metrics Collection
- **Search Performance**: Latency, throughput, accuracy
- **System Health**: CPU, memory, GPU utilization
- **Error Tracking**: Failed searches, model errors
- **Usage Analytics**: Query patterns, popular searches

### Alerting
- **Performance Degradation**: Slow search responses
- **System Failures**: Service unavailability
- **Data Quality**: Embedding generation failures
- **Capacity Planning**: Resource utilization thresholds

## Development Workflow

### Local Development
1. **Environment Setup**: Python virtual environment with dependencies
2. **Data Preparation**: Sample dataset with representative influencer data
3. **Model Download**: CLIP model weights and tokenizer
4. **Testing**: Unit tests for each component

### Deployment Pipeline
1. **CI/CD**: Automated testing and deployment
2. **Staging Environment**: Full system testing with production-like data
3. **Production Deployment**: Blue-green deployment for zero downtime
4. **Rollback Strategy**: Quick rollback to previous versions

## Technology Stack Summary

| Layer | Technology | Purpose |
|-------|------------|---------|
| API | FastAPI | REST API and web interface |
| Framework | LlamaIndex | Search infrastructure |
| ML Model | CLIP (OpenAI) | Multimodal embeddings |
| Vector DB | FAISS + LlamaIndex | Similarity search |
| Metadata DB | PostgreSQL/SQLite | Structured data |
| Cache | Redis | Query and result caching |
| Queue | Celery (optional) | Background processing |
| Monitoring | Prometheus + Grafana | Metrics and alerts |
| Deployment | Docker + K8s | Containerization | 