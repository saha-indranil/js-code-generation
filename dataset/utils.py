import os
from common import constants
import json
from functools import lru_cache
from tree_sitter import Language, Parser
import os
from collections import defaultdict
import re
import warnings

# ===== xxxxx ===== xxxxx ===== xxxxx ===== xxxxx ===== xxxxx ===== xxxxx ===== xxxxx ===== xxxxx

SUPPORTED_LANGUAGES = [
    constants.PYTHON, 
    constants.TYPESCRIPT, 
    constants.GO,
    constants.RUST, 
    constants.RUBY,
    constants.REACT_TS,
    constants.SWIFT,
    constants.MATLAB,
    constants.JAVASCRIPT,
    constants.JAVA,
    constants.CPP,
    constants.CSHARP,
    constants.PHP,
    constants.SCALA,
    "javascriptreact"
]

TREE_SITTER_LANGUAGE_NAME_MAPPING = {
    constants.PYTHON: constants.PYTHON,
    constants.TYPESCRIPT: constants.TYPESCRIPT,
    constants.GO: constants.GO,
    constants.RUST: constants.RUST,
    constants.RUBY: constants.RUBY,
    constants.SWIFT: constants.SWIFT,
    constants.MATLAB: constants.MATLAB,
    constants.JAVASCRIPT: constants.JAVASCRIPT,
    constants.JAVA: constants.JAVA,
    constants.REACT_TS: "tsx",
    "javascriptreact": constants.JAVASCRIPT,
    constants.CPP: constants.CPP,
    # constants.CSHARP: "csharp",
    constants.PHP: constants.PHP,
    constants.SCALA: constants.SCALA,
    constants.COBOL: constants.COBOL
}

# ===== xxxxx ===== xxxxx ===== xxxxx ===== xxxxx ===== xxxxx ===== xxxxx ===== xxxxx ===== xxxxx

def get_generated_parser_path():
    return os.path.join(os.path.dirname(__file__), "..", "generated-artifacts", "tree-sitter-parsers")

def get_language_binging_path():
    return os.path.join(os.path.dirname(__file__), "language-binding.json")

def get_tree_sitter_queries_path():
    return os.path.join(os.path.dirname(__file__), "queries")    

# ==== xxxx ==== xxxx ==== xxxx ==== xxxx ==== xxxx ==== xxxx ==== xxxx ====

@lru_cache(maxsize=None)
def get_extension_language_map():
    path = get_language_binging_path()
    with open(path, 'r') as f:
        config_data = json.load(f)

    extension_lang_map = {
        extension: lang
        for lang, extension_list in config_data.items()
        for extension in extension_list
    }
    return extension_lang_map

def get_language_from_extension(extension):
    extension_lang_map = get_extension_language_map()
    if extension in extension_lang_map:
        return extension_lang_map[extension]
    else:
        raise ValueError(f"No language found for extension {extension}")

@lru_cache(maxsize=None)
def get_tree_sitter_lang(lang: str):
    tree_sitter_path = os.path.join(get_generated_parser_path(), f"tree-sitter-{lang}.so")
    if not os.path.exists(tree_sitter_path):
        raise ValueError(f"Tree sitter path {tree_sitter_path} not found")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parser_lang = Language(tree_sitter_path, TREE_SITTER_LANGUAGE_NAME_MAPPING[lang])
    
    return parser_lang

@lru_cache(maxsize=None)
def get_parser(lang: str):
    parser_lang = get_tree_sitter_lang(lang)
    parser = Parser()
    parser.set_language(parser_lang)
    return parser

def get_parse_tree(parser, code):
    assert isinstance(code, str) or isinstance(code, bytes)
    if isinstance(code, str):
        code = bytes(code, "utf8")
    try:
        tree = parser.parse(code)
        return tree
    except Exception as e:
        raise ValueError(f"Error parsing code: {e}")

def get_capture_patterns(lang: str):
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Language {lang} not supported")
    
    query_path = os.path.join(get_tree_sitter_queries_path(), f"{lang}.scm")
    if not os.path.exists(query_path):
        raise ValueError(f"Query path {query_path} not found")
    with open(query_path, 'r') as f:
        query_pattern = f.read()
    
    return query_pattern

def query_matching_for_code(
        extension, 
        code,
        get_nodes = False,
        patterns_to_match = None
    ):
    lang = get_language_from_extension(extension)
    tree_sitter_lang = get_tree_sitter_lang(lang)
    if patterns_to_match is None:
        patterns_to_match = get_capture_patterns(lang)

    parser = get_parser(lang)
    
    return extract_capture_patterns(
        code,
        parser,
        tree_sitter_lang,
        patterns_to_match,
        get_nodes
    )

def extract_capture_patterns(
    code,
    parser,
    tree_sitter_lang,
    patterns_to_match,
    get_nodes
    ):
    tree = get_parse_tree(parser, code)
    query = tree_sitter_lang.query(patterns_to_match)
    captures = query.captures(tree.root_node)

    all_captures_map = defaultdict(list)
    for node, capture_name in captures:
        if get_nodes:
            all_captures_map[capture_name].append(node)
        else:
            all_captures_map[capture_name].append(node.text.decode('utf-8'))

    return all_captures_map

# ==== xxxx ==== xxxx ==== xxxx ==== xxxx ==== xxxx ==== xxxx ==== xxxx ====

def remove_comments(code, extension):
    extension_level_comment_map = {
        "java": """
            (line_comment) @comments
            (block_comment) @comments
        """,
        "rs": """
            (line_comment) @comments
        """
    }

    lang = get_language_from_extension(extension)
    parser = get_parser(lang)
    tree_sitter_lang = get_tree_sitter_lang(lang)

    comment_query = extension_level_comment_map.get(extension, "(comment) @comments")
    while True:
        capture_patterns = extract_capture_patterns(
            code,
            parser,
            tree_sitter_lang,
            comment_query,
            True
        )
        if len(capture_patterns.get("comments", [])) > 0:
            code_bytes = code.encode('utf-8')
            capture_first_node = capture_patterns["comments"][0]
            start_byte, end_byte = capture_first_node.byte_range[0], capture_first_node.byte_range[1]
            code_bytes = code_bytes[:start_byte] + code_bytes[end_byte:]
            code = code_bytes.decode('utf-8')
        else:
            break
    code = re.sub(r'(\n{2,})', '\n\n', code)
    return code

# ==== xxxx ==== xxxx ==== xxxx ==== xxxx ==== xxxx ==== xxxx ==== xxxx ====