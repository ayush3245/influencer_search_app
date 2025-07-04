# Influencer Discovery Tool

An AI-powered influencer discovery application that enables semantic search across influencer data using multimodal embeddings. Built with [LlamaIndex](https://www.llamaindex.ai/) and CLIP for text and image understanding.

## Features

- **Semantic Search**: Find influencers using natural language queries like "find female influencer with curly hair"
- **Multimodal Understanding**: Search across text (bios, descriptions) and images (profile photos, content thumbnails)
- **Fast Retrieval**: Efficient embedding-based search for quick results
- **Extensible Data Sources**: Support for PostgreSQL, CSV, and Excel data formats

## Getting Started

First, setup the environment with uv:

> **_Note:_** This step is not needed if you are using the dev-container.

```shell
uv sync
```

Then configure your environment variables in the `.env` file:
- Set your `OPENAI_API_KEY` if using any LLM functionality
- Configure data directories and search parameters

Second, prepare your influencer data in the `./data` directory and generate embeddings:

```shell
uv run generate
```

Third, run the development server:

```shell
uv run fastapi dev
```

Then open [http://localhost:8000](http://localhost:8000) with your browser to start the influencer search interface.

To start the app optimized for **production**, run:

```
uv run fastapi run
```

## Configuration

You can configure embedding models and search parameters in [settings.py](app/settings.py). The application uses CLIP embeddings for multimodal search capabilities.

## Use Case

This tool is designed for influencer marketing agencies to efficiently discover and match influencers based on various criteria:

- **Visual characteristics**: Hair color, style, appearance
- **Content type**: Beauty, fitness, lifestyle, etc.
- **Demographics**: Age range, location, audience
- **Engagement metrics**: Follower count, engagement rate

Example search queries:
- "Find fitness influencers with high engagement"
- "Female beauty influencers with curly hair"
- "Lifestyle content creators aged 25-35"

You can test the search functionality through the web interface or API endpoint:

```bash
curl --location 'localhost:8000/api/chat' \
--header 'Content-Type: application/json' \
--data '{ "messages": [{ "role": "user", "content": "Find female influencers with curly hair who create beauty content" }] }'
```

## Data Format

The application expects influencer data with the following structure:
- **Text fields**: Name, bio, content description, demographics
- **Image fields**: Profile photo URLs, content thumbnail URLs
- **Metadata**: Follower count, engagement rate, platform information

## Learn More

- [LlamaIndex Documentation](https://docs.llamaindex.ai) - Learn about the underlying framework
- [CLIP Model](https://openai.com/research/clip) - Understand the multimodal embedding approach
- [Semantic Search Guide](https://docs.llamaindex.ai/en/stable/understanding/embeddings/) - Learn about embedding-based search
