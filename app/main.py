from openenv.core.env_server import create_app
from app.environment import MegaForgeEnv
from app.models import Action, Observation

app = create_app(MegaForgeEnv, Action, Observation, env_name="crossmill-megaforge")
