
from typing import Annotated, Literal 
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
import os
from autogen import ConversableAgent
import time
from typing_extensions import Annotated
import autogen
from autogen.cache import Cache
import json
import ast
import os
from pathlib import Path
import sys
from typing import List, Dict
from autogen import AssistantAgent





# Configuration for summarization agent
config_list = [{
    "model": "llama3.2",
    "api_type": "ollama",
    "client_host": "http://localhost:11434",
}]




llm_config = {
    "request_timeout": 600,
    "config_list": config_list,
    "temperature": 0,
    "seed": 43
}



code_analyst_agent = AssistantAgent(
    
    name="code_analyst_agent",
    system_message=
                    "Only use the function you have been provided with,\
                    if the function has been called previously,return the word 'TERMINATE',\
                    you are responsible to Extract all absolute and relative imports from the given file.\
                    Resolve the paths of imported modules to locate their source files,\
                    Extract the full source code of imported classes or functions from the resolved files.",
    llm_config=llm_config
)




code_summarizer_classifier_agent = AssistantAgent(
    
    name="code_summarizer_classifier_agent",
    system_message=
                    "Only use the function you have been provided with,\
                    if the function has been called previously,return the word 'TERMINATE',\
                    Your tasks include: Summarizing the given Python function or class in 4 sentences,\
                    Assigning it to one of the following categories: train, configs, utils, dataloader, evaluation, model,\
                    Ensuring the response format strictly follows: CATEGORY: <category_name>.",
    llm_config=llm_config
)





code_saver_agent = AssistantAgent(
    
    name="code_saver_agent",
    system_message=
                    "Only use the function you have been provided with,\
                    if the function has been called previously,return the word 'TERMINATE',\
                    Your tasks include: Saving the given Python function or class to a file in the specified directory.",
    llm_config=llm_config
)




user_proxy = autogen.UserProxyAgent(
    name = "userproxy",
    max_consecutive_auto_reply=1,
    human_input_mode="NEVER",
    # code_execution_config= {"workdir":"test"},
    system_message="reply TERMINATE if the task has been solved at full satisfaction of user\
        otherwise reply CONTINUE or the reason why the task has not been solved yet "
)



# Set project root and folder to scan
REPO_ROOT = Path("/home/zohreh/workspace/my_project/clone_repo/dinov2")
FOLDER_TO_SCAN = REPO_ROOT / "dinov2"/"logging"
OUTPUT_DIR = REPO_ROOT / "categorized_code"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)  # Ensure directory exists




# Mapping categories to filenames
CATEGORY_MAPPING = {
    "train": "train.py",
    "configs": "config.py",
    "utils": "utils.py",
    "dataloader": "dataloader.py",
    "evaluation": "eval.py",
    "model": "model.py"
}



# Libraries to ignore
IGNORED_LIBRARIES = {
    "os", "sys", "argparse", "logging", "functools", "math", "json", "torch", "numpy",
    "torch.nn", "torch.nn.parallel", "typing", "collections", "datetime", "time",
    "pathlib", "random", "enum", "copy", "warnings", "csv", "PIL", "io", "torchvision", "re", "socket"
}



### **Extract Imports**
@user_proxy.register_for_execution()
@code_analyst_agent.register_for_llm(description="extract imports")
def extract_imports(file_path: Path, repo_root: Path) -> List[Dict[str, str]]:
    """Extract absolute and relative imports from a Python file."""
    imports = []
    try:
        source = file_path.read_text()
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("dinov2"):
                        imports.append({"type": "module", "name": alias.name})

            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split('.')[0] not in IGNORED_LIBRARIES:
                    resolved_module = node.module
                    if node.level > 0:
                        current_module_path = file_path.relative_to(repo_root).with_suffix("")
                        current_module_parts = list(current_module_path.parts)[:-1]
                        relative_module_parts = current_module_parts[: len(current_module_parts) - node.level + 1]
                        if node.module:
                            relative_module_parts.append(node.module)
                        resolved_module = ".".join(relative_module_parts)

                    for alias in node.names:
                        imports.append({"type": "symbol", "module": resolved_module, "name": alias.name, "level": str(node.level)})
    except Exception as e:
        print(f" Error processing {file_path}: {e}")
    return imports




### **Resolve Import Paths**
@user_proxy.register_for_execution()
@code_analyst_agent.register_for_llm(description="resolve import paths")
def resolve_import_path(import_item: Dict[str, str], current_directory: str) -> str:
    
#     """Find the corresponding file or directory for an imported module or symbol."""
    module_name = import_item.get("name") if import_item["type"] == "module" else import_item.get("module")

    if not module_name:
        print(f"Skipping import due to missing module name: {import_item}")
        return None

    module_parts = module_name.split(".")

    #  Absolute Imports (e.g., `from dinov2.utils import config`)**
    if module_name.startswith("dinov2"):
        absolute_path = REPO_ROOT.joinpath(*module_parts)
        possible_paths = [
            absolute_path.with_suffix(".py"),  # Check if it's a standalone Python file
            absolute_path / "__init__.py"      # Check if it's a package with `__init__.py`
        ]

    else:
        # Relative Imports (e.g., `from .helpers import Logger`)**
        if import_item.get("level", 0) > 0:
            relative_parts = module_name.lstrip(".").split(".")
            possible_paths = [
                current_directory.joinpath(*relative_parts).with_suffix(".py"),
                current_directory.joinpath(*relative_parts, "__init__.py")
            ]
        else:
            possible_paths = []

    # Check if the resolved path exists**
    for path in possible_paths:
        if path.exists():
            return str(path)

    print(f"Could not resolve path for import: {import_item}")
    return None




## **Extract Class or Function Content**
@user_proxy.register_for_execution()
@code_analyst_agent.register_for_llm(description="extract class or function content")
def extract_class_or_function_content(file_path: Path, symbol_name: str = None) -> str:
#     """Extract class or function content if a specific symbol is imported, otherwise read the full module."""
    try:
        source = file_path.read_text()
        tree = ast.parse(source)

        extracted_content = {}

        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                extracted_content[node.name] = ast.unparse(node)

        if symbol_name:
            return extracted_content.get(symbol_name, f"{symbol_name} not found in {file_path}")
        
        return "\n\n".join(extracted_content.values())

    except Exception as e:
        return f"Error extracting content from {file_path}: {e}"
    
    
    
    

### **Summarize and Categorize**
@user_proxy.register_for_execution()
@code_summarizer_classifier_agent.register_for_llm(description="summarize and categorize")
def summarize_and_categorize_content(class_or_function_contents: Dict[str, str]) -> Dict[str, str]:
    """Summarizes and categorizes extracted functions/classes using LLM."""
    categorized = {}

    for name, content in class_or_function_contents.items():
        if not content.strip():
            continue

        try:
            prompt = (
                f"Summarize the following Python function or class strictly in 4 sentences "
                f"and determine its category: (train, configs, utils, dataloader, evaluation, model). "
                f"Only return the category name in this format: CATEGORY: <category_name>.\n\n"
                f"{content[:2500]}"
            )
            response = code_summarizer_classifier_agent.generate_reply(messages=[{"role": "user", "content": prompt}])

            category = "utils"
            if isinstance(response, dict) and "content" in response:
                for line in response["content"].split("\n"):
                    if line.strip().startswith("CATEGORY:"):
                        category = line.strip().split(":")[-1].strip().lower()
                        break

            category = category if category in CATEGORY_MAPPING else "utils"
            categorized[name] = category

        except Exception as e:
            print(f" Error categorizing content: {str(e)}")

    return categorized




# ### **Save Extracted Source Code in Correct Files**
# @user_proxy.register_for_execution()
# @code_saver_agent.register_for_llm(description="save source code")
# def save_source_code(class_or_function_contents: Dict[str, str], categorized: Dict[str, str]):
    
#     """Saves extracted functions/classes into their corresponding categorized files."""
#     for name, category in categorized.items():
#         file_path = OUTPUT_DIR / CATEGORY_MAPPING[category]
#         with open(file_path, "a", encoding="utf-8") as f:
#             if class_or_function_contents[name].strip():  
#                 f.write(f"# {name} (Category: {category})\n{class_or_function_contents[name]}\n\n{'-'*80}\n\n")
                
               

# ### **Main Execution**
if __name__ == "__main__":
    
    

# Initialize imports dictionary
    imports = {}
    resolved_paths = {}
    class_or_function_contents = {}
    file_paths = []

     # Iterate over Python files
    for file_path in FOLDER_TO_SCAN.rglob("*.py"):  
        # print(f" Processing: {file_path}")  # Debugging print
        # file_paths.append(file_path)
        # print(f"file_paths: {file_paths}")
        

        response_1 = user_proxy.initiate_chat(
        code_analyst_agent,
        message=(
        f"Call the function 'extract_imports' with inputs: file_path={str(file_path)} and REPO_ROOT={str(REPO_ROOT)}. "
        f"Return the extracted imports as a Python dictionary."
    )
    )

         # Debugging: Print full chat history
        print(f"Full Chat History for {file_path}:\n{response_1.chat_history}")

         # Process chat history
        for res in response_1.chat_history:
            if 'tool_responses' in res:
                extracted_imports = res['tool_responses'][0]['content']
                
                    
                # Debugging: Print extracted data for each file
                print(f"extracted_imports: {extracted_imports}")

                    #  Fix: Parse JSON response if needed
                if isinstance(extracted_imports, str):  
                    try:
                            extracted_imports = json.loads(extracted_imports)  # Convert from JSON string to list
                    except json.JSONDecodeError:
                        print(f"JSON decoding error for {file_path}: {extracted_imports}")
                        continue                                               # Skip this file if parsing fails

                    # Ensure response is a list before storing
                if isinstance(extracted_imports, list):  
                    imports[str(file_path)] = extracted_imports
                    # print(f"imports: {imports}")  
                else:
                    print(f"Unexpected format for {file_path}: {extracted_imports}")
                

    for py_file, imports_list in imports.items():
            
        
        for import_item in imports_list:
            
            current_directory = str(Path(py_file).parent)

            response_2 = user_proxy.initiate_chat(
            code_analyst_agent,
    
            message=(
            f"Call the function 'resolve_import_path' with inputs: "
            f"import_item={import_item} and current_directory='{current_directory}'. "
            f"Return the resolved import path."
        )
                )

                    # print(f"response_2.chat_history:{response_2.chat_history}")
            for res in response_2.chat_history:
                if 'tool_responses' in res:
                    resolved_path = res['tool_responses'][0]['content']
                    print(f"resolved_path: {resolved_path}")
                    if resolved_path:
                        response_3 = user_proxy.initiate_chat(
                            code_analyst_agent,
                            message=(
                                f"Call the function 'extract_class_or_function_content' with inputs: "
                                f"resolved_path={resolved_path}. "
                                f"Return the extracted class or function content."
                            )
                        )
                        for res in response_3.chat_history:
                            if 'tool_responses' in res:
                                extracted_content = res['tool_responses'][0]['content']
                                if extracted_content:
                                    class_or_function_contents[import_item["name"]] = extracted_content
                                    

#                         # response_4 = user_proxy.initiate_chat(
#                         #     code_summarizer_classifier_agent,
#                         #     message=(
#                         #         f"Call the function 'summarize_and_categorize_content' with inputs: "
#                         #         f"class_or_function_contents={class_or_function_contents}. "
#                         #         f"Return the summarized content."
#                         #     )
#                         # )
#                         # for res in response_4.chat_history:
#                         #     if 'tool_responses' in res:
#                         #         summarized_content = res['tool_responses'][0]['content']
#                         #         for name, category in summarized_content.items():
#                         #             print("\n" + "="*80)
#                         #             print(f" **Function/Class Name:** {name}")
#                         #             print("="*80 + "\n")

#                                 # response_5 = user_proxy.initiate_chat(
#                                 #     code_analyst_agent,
#                                 #     message=(
#                                 #         f"Call the function 'save_source_code' with inputs: "
#                                 #         f"class_or_function_contents={class_or_function_contents}, summarized_content={summarized_content}. "
#                                 #         f"Return the summarized content."
#                                 #     )
#                                 # )
#                                 # for res in response_5.chat_history:
#                                 #     if 'tool_responses' in res:
#                                 #         saved_content = res['tool_responses'][0]['content']
#                                 #         print(f"Saved Content: {saved_content}")

                   

# # 
#     # # print(f"imports: {imports}")
#     # # print(f"class_or_function_contents: {class_or_function_contents}")

#     # categorized = summarize_and_categorize_content(class_or_function_contents)
    
      
#     # # Display structured output
#     # for name, category in categorized.items():
#     #     print("\n" + "="*80)
#     #     print(f" **Function/Class Name:** {name}")
#     #     print(f" **Category:** {category.capitalize()}")
#     #     print(f" **Source Code:**\n{class_or_function_contents[name]}")
#     #     print("="*80 + "\n")

#     # # Save categorized functions/classes
#     # save_source_code(class_or_function_contents, categorized)

#     # print("\n Categorized functions/classes saved into correct files.")











