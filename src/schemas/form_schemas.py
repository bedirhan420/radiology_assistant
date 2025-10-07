from pydantic import BaseModel,Field
from typing import Optional,List

class PatientInfos(BaseModel):
    name:Optional[str] = Field(None,description="Name of the patient")
    surname:Optional[str] = Field(None,description="Surname of the patient")
    age:Optional[int] = Field(None,description="Age of the patient")
    id:Optional[str] = Field(None,description="TC identity number of the patient")

class FormMRI(BaseModel):
    patient_informations:PatientInfos=Field(description="Informations of the relevant patient")
    mr_area:str=Field(description="the body part where the MRI was taken. eg: 'Brain','Servical'")
    findings:str=Field(description="Details of the medical findings stated in the report.")
    result:str=Field(description="The final conclusion or diagnosis reached by the doctor or radiologist")
    dr_note:Optional[str]=Field(None,description="Additional doctor's notes, if any.")

class FormBloodTest(BaseModel):
    patient_informations:PatientInfos=Field(description="Informations of the relevant patient")
    hemoglobin:Optional[float]=Field(description="Hemoglobin value (HGB) (g/dl)")
    wbc:Optional[float]=Field(description="White blood cell (WBC) count (K/uL)")
    platelet:Optional[int]=Field(description="Trombosit (PLT) count (K/uL)")
    dr_note:Optional[str]=Field(None,description="Additional doctor's notes, if any.")

class PatientTextChunk(BaseModel):
    patient_name: str = Field(description="The full name or a clear identifier of the detected patient (e.g., 'Ahmet Yılmaz').")
    related_text: str = Field(description="The combined text of all sentences from the entire transcript that relate only to this patient.")

class TranscriptAnalysis(BaseModel):
    patient_chunks: List[PatientTextChunk]