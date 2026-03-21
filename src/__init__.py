import logging
import dagshub
from dotenv import load_dotenv

# Environment
load_dotenv()

# Initialize DagsHub with Credentials
dagshub.init(
    repo_owner="Ronizorzan",
    repo_name="credit-default-pipeline"
)

# Configure the logging strategy
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler()
    ]
)


