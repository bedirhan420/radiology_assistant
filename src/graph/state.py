from typing import TypedDict,Optional
from config import FORM_TYPE

class GraphState(TypedDict):
    patient_name:str
    text_chunk:str
    transcript:Optional[str]
    form_type:FORM_TYPE
    extracted_data:Optional[dict]
    error:Optional[str]