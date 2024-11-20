import os
from tree_sitter import Language
import tree_sitter_utils.utils as tsutils
from common.constants import (TYPESCRIPT, REACT_TS)

# ===== xxxxx ===== xxxxx ===== xxxxx ===== xxxxx ===== xxxxx ===== xxxxx ===== xxxxx ===== xxxxx

def setup_treesitter():
    tree_sitter_path = tsutils.get_generated_parser_path()
        
    os.makedirs(tree_sitter_path, exist_ok=True)
    os.system(f"cp -r /ts-parsers/* {tree_sitter_path}/")
    return

    # Some issue executing this on my VM, downloading from pre-computed parsers.
    def _build_library(parser_path, grammer_path, additional_setup_fn = None):
        if os.path.exists(parser_path):
            return
        if additional_setup_fn:
            additional_setup_fn()

        Language.build_library(
            parser_path,
            [grammer_path]
        )

    tree_sitter_repos = [
        "https://github.com/tree-sitter/tree-sitter-typescript.git",
        "https://github.com/tree-sitter/tree-sitter-rust",
        "https://github.com/tree-sitter/tree-sitter-python.git",
        "https://github.com/tree-sitter/tree-sitter-go",
        "https://github.com/tree-sitter/tree-sitter-ruby",
        "https://github.com/alex-pinkus/tree-sitter-swift",
        "https://github.com/acristoffers/tree-sitter-matlab",
        "https://github.com/tree-sitter/tree-sitter-javascript.git",
        # "https://github.com/tree-sitter/tree-sitter-java"
    ]
    tree_sitter_path = tsutils.get_generated_parser_path()
    os.makedirs(tree_sitter_path, exist_ok=True)

    for tree_sitter in tree_sitter_repos:
        repo_name = tree_sitter.split("/")[-1].split(".")[0]
        # todo: might not be valid for some langs like c-sharp
        lang = repo_name.split("-")[-1]

        if not os.path.exists(os.path.join(tree_sitter_path, repo_name)):
            os.system(f"git clone {tree_sitter} {tree_sitter_path}/{repo_name}")
        
        if lang=="swift":
            # Build the repo
            grammer_path = f"{tree_sitter_path}/{repo_name}"
            parser_path = f"{tree_sitter_path}/tree-sitter-{lang}.so"
            
            def _setup_fn():
                exit_code = os.system(f"cd {grammer_path} && npm install")
                if exit_code!= 0:
                    raise ValueError(f"Error building swift repo {grammer_path}")
            
            _build_library(parser_path, grammer_path, _setup_fn)

        elif lang=="typescript":
            ts_like_paths = {
                "typescript": TYPESCRIPT,
                "tsx": REACT_TS
            }
            for ts_like_path, lang_name in ts_like_paths.items():
                grammer_path = f"{tree_sitter_path}/{repo_name}/{ts_like_path}"
                parser_path = f"{tree_sitter_path}/tree-sitter-{lang_name}.so"
                _build_library(parser_path, grammer_path)   
        else:
            grammer_path = f"{tree_sitter_path}/{repo_name}"
            parser_path = f"{tree_sitter_path}/tree-sitter-{lang}.so"
            _build_library(parser_path, grammer_path)

if __name__ == "__main__":
    setup_treesitter()