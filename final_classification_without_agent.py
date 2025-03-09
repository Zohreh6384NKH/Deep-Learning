

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



# Define the summarization agent
summarization_agent = AssistantAgent(
    name="summarization_agent",
    system_message="Summarize the provided Python function or class in 4 sentences and categorize it into: "
                   "(train, configs, utils, dataloader, evaluation, model). "
                   "Return the category name in this format: CATEGORY: <category_name>.",
    llm_config=llm_config
)




# Set project root and folder to scan
REPO_ROOT = Path("/home/zohreh/workspace/my_project/clone_repo/dinov2")
# FOLDER_TO_SCAN = REPO_ROOT / "dinov2"/"eval"/"depth"/"models"/"backbones"
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
                        imports.append({"type": "symbol", "module": resolved_module, "name": alias.name, "level": node.level})
    except Exception as e:
        print(f" Error processing {file_path}: {e}")
    return imports




### **Resolve Import Paths**
def resolve_import_path(import_item: Dict[str, str], current_directory: Path) -> Path:
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
            absolute_path / "__init__.py"  # Check if it's a package with `__init__.py`
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
            return path

    print(f"Could not resolve path for import: {import_item}")
    return None




### **Extract Class or Function Content**
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
def summarize_content(class_or_function_contents: Dict[str, str]) -> Dict[str, str]:
    """Summarizes and categorizes extracted functions/classes using LLM."""
    categorized_imports = {}

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
            response = summarization_agent.generate_reply(messages=[{"role": "user", "content": prompt}])

            category = "utils"
            if isinstance(response, dict) and "content" in response:
                for line in response["content"].split("\n"):
                    if line.strip().startswith("CATEGORY:"):
                        category = line.strip().split(":")[-1].strip().lower()
                        break

            category = category if category in CATEGORY_MAPPING else "utils"
            categorized_imports[name] = category
            print(f"Category : {category}")

        except Exception as e:
            print(f" Error categorizing content: {str(e)}")

    return categorized_imports




### **Save Extracted Source Code in Correct Files**
def save_source_code(class_or_function_contents: Dict[str, str], categorized: Dict[str, str]):
    """Saves extracted functions/classes into their corresponding categorized files."""
    for name, category in categorized.items():
        file_path = OUTPUT_DIR / CATEGORY_MAPPING[category]
        with open(file_path, "a", encoding="utf-8") as f:
            if class_or_function_contents[name].strip():  
                f.write(f"# {name} (Category: {category})\n{class_or_function_contents[name]}\n\n{'-'*80}\n\n")
                
                
                
                
                


### **Main Execution**
if __name__ == "__main__":
    
    print(f"\n Scanning repository: {FOLDER_TO_SCAN}")

    imports = {file_path: extract_imports(file_path, REPO_ROOT) for file_path in FOLDER_TO_SCAN.rglob("*.py")}
    
    class_or_function_contents = {}
    for py_file, import_list in imports.items():
        for import_item in import_list:
            resolved_path = resolve_import_path(import_item, py_file.parent)
            print(f"import_item: {import_item}")
            print(f"py_file.parent: {py_file.parent}")
            print(f"resolved_path: {resolved_path}")
            if resolved_path:
                # extracted_content = extract_class_or_function_content(resolved_path, import_item["name"])
                extracted_content = extract_class_or_function_content(resolved_path)
                if extracted_content:
                    class_or_function_contents[import_item["name"]] = extracted_content
                    
    # print(f"imports: {imports}")
    # print(f"resolved_path: {resolved_path}")

    # print(f"class_or_function_contents: {class_or_function_contents}")

    categorized_imports = summarize_content(class_or_function_contents)
    print(f"categorized: {categorized_imports}")
    
    #Display structured output
    for name, category in categorized_imports.items():
        print("\n" + "="*80)
        print(f" **Function/Class Name:** {name}")
        print(f" **Category:** {category.capitalize()}")
        print(f" **Source Code:**\n{class_or_function_contents[name]}")
        print("="*80 + "\n")

    # Save categorized functions/classes
    save_source_code(class_or_function_contents, categorized_imports)

    print("\n Categorized functions/classes saved into correct files.")











