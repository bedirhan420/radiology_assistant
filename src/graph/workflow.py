from langgraph.graph import StateGraph,END
from graph.state import GraphState
from graph.nodes import *

def should_contiune(state:GraphState)->GraphState:
    ''' router nodedan sonra hangi yolu seçeceğine karar verir.'''
    if state["form_type"] =="mri":
        return "extract_mri"
    elif state["form_type"]=="blood_test":
        return "extract_blood_test"
    else:
        return "handle_error"
    
graph = StateGraph(GraphState)

graph.add_node("router",node_router)
graph.add_node("extract_mri",node_extract_mri)
graph.add_node("extract_blood_test",node_extract_blood_test)
graph.add_node("handle_error",node_handle_error)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    should_contiune,
    {
        "extract_mri":"extract_mri",
        "extract_blood_test":"extract_blood_test",
        "handle_error":"handle_error",        
    }
)

graph.add_edge("extract_mri",END)
graph.add_edge("extract_blood_test",END)
graph.add_edge("handle_error",END)

app = graph.compile()