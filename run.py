from hypercorn.config import Config
from hypercorn.asyncio import serve
import asyncio
from app import create_app

app = create_app()

if __name__ == '__main__':
    config = Config()
    config.bind = ["localhost:5001"]
    asyncio.run(serve(app, config)) 