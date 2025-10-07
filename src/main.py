import json
from config import AUDIO_DIR,OUTPUT_DIR
from graph.workflow import app

def process_audio_files():
    AUDIO_DIR.mkdir(parents=True,exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    
    print(f"\n{'-'*50}\nWorkflow with Agent is starting...\n{'-'*50}")

    audio_files = [f for f in AUDIO_DIR.iterdir() if f.is_file() and not f.name.startswith('.')]
   
    if not audio_files:
        print(f"No audio files were found to process. Please add your audio files to the '{AUDIO_DIR}' folder.")
        return
    
    for f in audio_files:
        print(f"\n>>> Processing File: {f.name}")
        inputs = {"audio_path":str(f)}
        final_state = app.invoke(inputs)
        
        output_file_name = OUTPUT_DIR / f"{f.stem}.json"
        
        output_data = {
            "source_file": f.name,
            "transcript": final_state.get("transcript"),
            "determined_form": final_state.get("form_type"),
            "extracted_data": final_state.get("extracted_data"),
            "error": final_state.get("error")
        }
        
        with open(output_file_name,'w',encoding="utf-8") as f:
            json.dump(output_data,f,ensure_ascii=False,indent=4)
            
        print(f"<<< Result saves the '{f.name}' file. ")

    print(f"\n{'-'*50}\nProcess is done.\n{'-'*50}")
    
if __name__ == "__main__":
    process_audio_files()