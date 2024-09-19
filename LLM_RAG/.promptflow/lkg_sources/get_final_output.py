from promptflow import tool
import json

# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
 
@tool
def get_final_output(user_question: str, rag_response: str, source_documents: list) -> dict:  
    output = {
            "user_question": user_question,
            "response": rag_response,
            "source_documents": source_documents
    }
 
    output_json = json.loads(json.dumps(output))
 
    return output_json