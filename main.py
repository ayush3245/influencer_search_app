import logging

from app.settings import init_settings
from app.workflow import create_workflow
from app.data_init import ensure_data_loaded, get_data_status
from dotenv import load_dotenv
from llama_index.server import LlamaIndexServer, UIConfig

logger = logging.getLogger("uvicorn")

# A path to a directory where the customized UI code is stored
COMPONENT_DIR = "components"


def create_app():
    app = LlamaIndexServer(
        workflow_factory=create_workflow,  # A factory function that creates a new workflow for each request
        ui_config=UIConfig(
            component_dir=COMPONENT_DIR,
            dev_mode=True,  # Please disable this in production
            layout_dir="layout",
        ),
        logger=logger,
        env="dev",
    )
    # You can also add custom FastAPI routes to app
    app.add_api_route("/api/health", lambda: {"message": "OK"}, status_code=200)
    
    # Data status endpoint
    app.add_api_route("/api/data/status", get_data_status, status_code=200)
    
    # Debug endpoint to verify loaded influencers
    def debug_influencers():
        from app.search_engine import get_search_engine
        search_engine = get_search_engine()
        results = search_engine.search_text("influencer", limit=50)
        
        return {
            "total_influencers": len(results),
            "influencers": [
                {
                    "name": r.influencer.name,
                    "category": r.influencer.category,
                    "bio": r.influencer.bio,
                    "followers": r.influencer.follower_count
                }
                for r in results
            ]
        }
    
    app.add_api_route("/api/debug/influencers", debug_influencers, status_code=200)
    
    # Initialize data on startup
    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting application and initializing data...")
        try:
            init_result = ensure_data_loaded()
            logger.info(f"Data initialization completed: {init_result}")
            
            if init_result.get('status') == 'failed':
                logger.error("Failed to load any data - application may not function correctly")
            else:
                stats = init_result.get('stats', {})
                total_influencers = stats.get('total_influencers', 0)
                logger.info(f"Application ready with {total_influencers} influencers loaded")
                
        except Exception as e:
            logger.error(f"Error during data initialization: {e}")
    
    return app


load_dotenv()
init_settings()
app = create_app()
