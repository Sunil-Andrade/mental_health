from pydantic import BaseModel
from typing import List, Dict

class Data(BaseModel):
    user_id: str
    history: List[Dict]
    current: Dict