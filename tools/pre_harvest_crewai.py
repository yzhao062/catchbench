"""Harvest unlabeled PRE staging rows from public CrewAI projects."""
from __future__ import annotations

import argparse
import ast
import json
import re
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "pre_staging" / "crewai.json"

SOURCE_REPOS: tuple[tuple[str, str, str], ...] = (
    ("liangdabiao/crewai_stock_analysis_system", "aab66c6a5782bb74542bdce6996baba30742e337", "NOASSERTION"),
    ("opahopa/crewai-factory-crew", "2dbcb84e0abeb21b15c9c53115abffd9d0d6e96f", "GPL-3.0"),
    ("tylerprogramming/travel-planner", "1ae200daa65608d700b252fd7da748667e63312d", "NOASSERTION"),
    ("ctreminiom/cv_optimizer", "c889d81dbc92802ef86bd25e66017d5a824daf6b", "MIT"),
    ("HopMaster03/linkedin-post-generation-CrewAI", "08300406d9184887aedda5cf20e83ebbb2513845", "MIT"),
    ("Mohankumar217/CrewAi", "9ba08661e14a784ad660cd11be963d698db5ca3f", "MIT"),
    ("rchow93/askvai-meta-agentic-agents", "1a2fd9ebdb0ede411f15e370a43132b9a77ffafe", "Apache-2.0"),
    ("Amteshwar091/crewAI-multi-agent-marketing-system", "a40d9912d9d2847e118182ab7fbd3b44be831439", "MIT"),
    ("aniket-work/How-I-Built-Scrum-Master-AI-Agent", "e907af31833744c049311c36550d71435ebc2a42", "Apache-2.0"),
    ("lausantosdev/gerador-ebook-crewai", "8932aa4cefa33d5790bc46cd528274eeb6301ea0", "NOASSERTION"),
    ("rohitgarwad/crewai-agenticai", "4e6cea76d7536f4eae21fb1e2036727dbaafb790", "NOASSERTION"),
    ("sudeepk0714U/CREW_AI", "9cf949882a6c8a8e74715c3283f762f22590039b", "NOASSERTION"),
    ("Tolosa527/weekly-report-assistant", "fa4ecc0cfe686297d89a8b74e012fa770eb1d110", "Apache-2.0"),
    ("sobit-nep/crewAI-agents", "477c59067470b945210b6e3a60571f1d4af986ca", "NOASSERTION"),
    ("ItsAli1711/crewAi_test", "7bf52c21de58db326f05357c1d407abf32c53fe3", "NOASSERTION"),
    ("manoharjakkampudi/crew-ai-post", "954f30c3218d59fb9644af8b0f7e23c6c6180806", "NOASSERTION"),
    ("jinyoung/process-gpt-generic-agent", "a1cfb9a26d81d4cee955127fdcde1a2a5b803a80", "NOASSERTION"),
    ("bishwast/Agentic-SOC", "58789c3c6f284ad6d18dadc82f6e0c143dce383c", "NOASSERTION"),
    ("durenajafamjad/crewai-adversarial-evaluation", "6486cc94a0cfd3d5d8b7bb8f98ac95ccfafbee3f", "NOASSERTION"),
    ("nallagondu/stock_market_genticAI", "166d0ba5e053b6230d2fa647ea0335fb91412a20", "NOASSERTION"),
    ("shashankyadav03/ai-property-researcher", "5aaa60f5ee70646d9a286ce20475833ddc3a60bd", "NOASSERTION"),
    ("DeividiJaeger/crewai-wiki-agent", "03d2b22e8470780f73d747dd380b192279d6019b", "NOASSERTION"),
    ("EngBaz/Multi-Agent-Financial-Analyst", "f52c691f8b143fd983969693ada745c79efda368", "NOASSERTION"),
    ("shawn1207/finance-alchemy", "9f88e78954d3dcbc4896e496a82cda75c92c9ab6", "NOASSERTION"),
    ("gurezende/Crew_Writer", "0e0c4358ece3abb2a90c43e12498eaa142be584e", "NOASSERTION"),
    ("kavindugit/crewai-tutorial", "5b56084feb2d88714cb80d34a47eb35395efb0ee", "NOASSERTION"),
    ("Coldvoltt/auto_analytics", "aafaf8f86b84a2116354feb1bf5917eb31fe9321", "Apache-2.0"),
    ("ssvasan369/travel-planner", "f9810a6d6d3eefd3ff12d73ae1fcb1941fc78ad9", "NOASSERTION"),
    ("ulfimlg/LinkedInPostGeneraterCrew", "fa83e13502e8d2545c5eda37c68f6012284c41e1", "MIT"),
    ("Amteshwar091/autonomous-marketing-agency", "f3ddef95a6658b0d60efafccf88be1e18b97fe87", "MIT"),
    ("aniket-work/How_I_Trained_AI_Agents", "b1c30866d24a2c0336c8e44ddabe60709fb76a4b", "Apache-2.0"),
    ("iainmckirdy/product-agents", "569803f6030c9ea5bbab251d16aa347aa0690944", "MIT"),
    ("JoseLFernandez/crewai_interface", "b71e6770d4b994b923dc33ec62c888d11deea0dd", "NOASSERTION"),
    ("OssiRuhanen/CrewAI_Assistant", "419306f0bdfa6080c92430835ff1cc1292949bf4", "NOASSERTION"),
    ("NazmulHudaNabil/Timezz-Facebook-Automation-with-CrewAI", "dee53059c7ad6a465fbc793504a7b8f92d24f056", "MIT"),
    ("hadiahameed/multi-agent-analysts", "768fcb2a5acf145b99b1e138a6f2da0fd41317f2", "NOASSERTION"),
    ("rafid29mehda/CrewAI-Automated-Stock-Analysis-System", "b37e221d35dd0af1e14c125a65049c073193760d", "NOASSERTION"),
    ("kareem1207/Bravo", "983833cdf18570ae7af95af577a277a5571da81d", "NOASSERTION"),
    ("victorcastillojimenez/CV-Scan", "198e4378b6eefd8a51d8a0a6c7a6a2a8bc611824", "NOASSERTION"),
    ("bmh2127/pd-discovery-platform", "c2325785aacc8dab85aa624d9be4cb515326d1f6", "MIT"),
    ("pandemonium0225/dynamic-coding-system", "95da3c94f8fd8d004dafb6c48352b15501b19a9a", "NOASSERTION"),
    ("mshariqa/stock_picker_crewai", "f16e64dc6ffebbfe9bdbd7d58bab286238193ba1", "NOASSERTION"),
    ("LikhithAvinash/Podcast_Generator", "d9d267d4b2b166c1eee4eae99ec1a001fc6d0c5a", "NOASSERTION"),
    ("saitanay/LLMageddon", "2616ed475e102a169b0b6c49a88d57447f80ec2a", "NOASSERTION"),
    ("YUGESHKARAN/Mentor-Consulting-Crew", "c9bb5ff269f57bee46e619f57697cbe3973b6fe0", "MIT"),
    ("AryanneGoncalves/AgentsJC", "10caf0ae7b3204fd03a2cc8f74cba359d7d948ed", "NOASSERTION"),
    ("Fall2Rise/Earnetics-", "4e8e5eb64fbf56f7715bf97be3ac1fed8b7cd362", "NOASSERTION"),
    ("TryCatchRaunak/reactwebsitemaker", "885545947dcc60ac238189dec11e93884833757b", "NOASSERTION"),
    ("pverhaert/crewai_course-builder", "2ff4b6c45226f6e7725595fa6982acc55efde8ba", "NOASSERTION"),
    ("korjavin/crewai-observability", "0565e77ca46de593ce9f55e94ee2f43bb06aacd0", "NOASSERTION"),
    ("giacomomiceli/deeplearningai_multiAiAgentSystems", "7b587cf8f524df714ff59080d3be9c7b2e9491dd", "NOASSERTION"),
    ("herissonsilvahs/smart_games_search_poc", "0a45c3301c4eb08c3fc953729dcf93b684e22d0c", "NOASSERTION"),
    ("HaseebUllahAbbasi/document-reader-using-rag-with-limitation", "87dd9e5aeab2ab648ff409c433ae058b2e0a08f9", "NOASSERTION"),
    ("AkshayParab1605/API-Healthcheckup-QA-Warroom", "094806ab560f9e17a1b467b3462b42b352b38b28", "NOASSERTION"),
    ("ksriyesh/h1b_smart_job_matcher", "8d06f38f0b5feb0b3d401ecfffedd288cccdb2d5", "NOASSERTION"),
    ("OneDuckyBoy/marketing-MAS-multi-agent-system", "9d3de04f72ff98d3066959d503348a00c5a40542", "MIT"),
    ("botextractai/ai-crewai-multi-agent", "83451d29ec6638c3eb7611ddf5ea62870f6686f8", "NOASSERTION"),
    ("codebasics/crewai-crash-course", "ebd3172f13f4820f312ea83ff242f9db5fb49814", "NOASSERTION"),
    ("HaileyTQuach/Smart-Nutritional-App", "a28f7da36d56929b092fbd48836b282f8d5f098b", "NOASSERTION"),
    ("Bhavik-Jikadara/youtube-automation-agent", "5ccb3f59e37bf7f2405e02dd38e99926d8a75375", "NOASSERTION"),
    ("UtsavTomar/crewai123", "4dad3ed50a2d2225ae7f74867a3eede538f0a1e9", "NOASSERTION"),
    ("FamilOrujov/financial-researcher-agentic-ai", "805b35284d4966181d94ed41a88bdd7b5f234837", "NOASSERTION"),
    ("Bhatteryash/E-commerce-Multi-Agent", "64d8f4b4b32c8abab11835fc02b09ab426248cdb", "NOASSERTION"),
    ("gabrielmarcolino23/crewai-services", "9c6dbf99a47d50c022f2ff15c70c3d2bc19f87fa", "NOASSERTION"),
    ("godoftheduckplayers/crewai", "a68f39c9bcf149bafdc8b05671672b2629192db1", "NOASSERTION"),
    ("diya04b/news_scraper", "b5505c839ae8d3c5961fb2cf3f72d58385d69adc", "NOASSERTION"),
    ("Siddhu83/traffic_optimization", "0701d3c90fb76e1cf8394913264b7a5634acb442", "NOASSERTION"),
    ("fredzolio/sms_public_health_monitor", "5d1994ecbf8430770cf711f6269cd8236ecd5c9e", "NOASSERTION"),
    ("Tejas-Bantupalli/trading_knowledge_base", "09417379f8cd379d7cd58ed30c7721cc14a509dc", "NOASSERTION"),
    ("andreagroferreira/myAIDEVTEAM", "f31b9d2c5a468fd757450448eb5fb6265dd7896c", "NOASSERTION"),
    ("tom333/cv", "a4696394614ff49f4a3099aa806b7a061ddd86d8", "GPL-3.0"),
    ("SANNNNN-123/youtube_summarizer", "85471eada66bae3042ac28bd05b7f6bc9df959a7", "NOASSERTION"),
    ("anshxl/vibecheck", "eb7ec990ba6cf9c9752f71d8db7e4dce717fdbba", "NOASSERTION"),
    ("yurifilgueira/seap_ai_agent_with_crew_ai", "f9a1b7b64b7693472956af8144583d4615edfaa2", "NOASSERTION"),
    ("ahmad-raza-4/skill-sieve", "27637b0026e1c98d225b3c3e5b21d414232190cf", "NOASSERTION"),
    ("italofarve/stock-analysis", "a2e9cf2d5d4a509a0803ccba51709f3a12dfc3a5", "NOASSERTION"),
    ("Surbhit01/Study_Companion", "4f90a55d0db77a92ace91b1a493de187ec434ead", "NOASSERTION"),
    ("Subrahmanyam2305/claude_computer_use", "0aa69e7c165b3193201d9a00edaeb0d7cc53b8e5", "MIT"),
    ("Saleh7127/multi-agent-masters-thesis-finder", "51b54e6a7061669673e269e9fa5a280aa81b5a08", "MIT"),
    ("Unica2804/Vaultmind", "4e8e61af2302d6afe960165c757ece5c22b22197", "NOASSERTION"),
    ("Thiagovilela2001/Crew_ai_v1", "ad35a89b86cc3c69f08ca91c76eda367a6ede463", "NOASSERTION"),
    ("DavidGCalles/lifeOS", "3d03bf48ffb3aaca1f788bf5ee9bbf1f4b874818", "NOASSERTION"),
    ("Vipul111196/stock-analysis-crewai", "899f29dc6225a77cf747dfc382d160ed3b47bc19", "NOASSERTION"),
    ("Sudip-8345/MASCRISS-AI", "86c2cb127d69d70bc9154b3edb242dcbe654fa5f", "MIT"),
    ("Dudiesz/synapse-weekly", "d5e9a539bedc2f4ca0b90f4e13e2e527965d5cee", "NOASSERTION"),
    ("mosh1331/stock-predicting-ai-agents", "4bd3036e5e5d003c6a1247c9ea47d6b458c42887", "NOASSERTION"),
    ("AravindB98/Cerebro", "54dda77852eb9206b9cd4252b0df76029bbeb21e", "NOASSERTION"),
    ("tony3liu/Auto_edit_test_cases", "b19a0e43d5af681faf27be2653b75a7e739e72b9", "NOASSERTION"),
    ("barackm/conekthub", "ade3490cd57be558c1aafcb2a277af368666ee5c", "NOASSERTION"),
    ("GeorgeMyller/resume_optimizer_crew_v2", "a0c465332b67350035511dd1c3ca46bedfb06039", "NOASSERTION"),
    ("bnarasimha21/crewai-newsletter-generator", "c70d36f6728f24334ecec2687860f8036c96c001", "MIT"),
    ("pryyyynz/DevCrew-Agents", "3f0a85bcf9d7e7847d3c7e960705f6d61f8bdfbf", "MIT"),
    ("xezbeth/Web-Researcher-Agent", "0a5a01d65f7a1307cd4821c65dab37f742e06d73", "NOASSERTION"),
    ("zcaceres/agentstack-receipts-manager", "0582390206f121abee2610f12993eb2642e4a8bc", "MIT"),
    ("Sanket22g/Myagent_ai_news", "fb126ff74aa78605f2521dfbf01903f1981404cb", "NOASSERTION"),
    ("armandogon94/Portfolio-AI-Agents", "395394da07ab2e41ada648d4ce6f5e5e93ac0a14", "NOASSERTION"),
    ("nandhana31/Roam.io", "83449b9a9ace4c626e782a7ca4ab473b3fb391db", "NOASSERTION"),
    ("farhafahmi/LinkedIn-content-agent", "9cc5d2c35b08eb992634cd80fc0a3e636880f44b", "NOASSERTION"),
    ("Mzafeer11/NourishBot-with-CrewAI", "b3e8005321b922841085a1c28fe35e88ae975167", "NOASSERTION"),
    ("Mira-dmh/CrewAI", "97733014f613430eed21d9f5e69423aebe9367b0", "NOASSERTION"),
    ("priyanshiiitr/AI_Nutrition_Coach", "48c2d17d2c8ba30de4316dc64374972ce6142b34", "NOASSERTION"),
    ("JasperHG90/rfcrew", "ec1539c4595b5a24a9aa940ce9aaf324a9c32928", "NOASSERTION"),
    ("jasonssdev/dl-crewai", "bd4d10f9e4e59fa76e99a6e9b015ca3a1584315c", "MIT"),
    ("talyssonoliver/sota", "1a3f79fe8fe4fb7fe5920031be93a5fc1710fd5b", "MIT"),
    ("kvrancic/interview-analyzer-archiver", "932c38b31b020db6cc6b3b1834e5e85ce1b25925", "NOASSERTION"),
    ("arpankumarde/crew-ai-multi-agent-orchestration-support-team", "32e2b3323d27fd7e0d3feb292f4c60d8cece6aeb", "NOASSERTION"),
    ("SomeOrdinaryDEV/infrastructure-sentiment-analysis", "6f13afb74b7f4fac12a282e61f6cd1011ebc1ca8", "NOASSERTION"),
    ("garvit5555/transportation_service_chatbot_using_agent", "70e6448626a10fe7f4c96d15c6b6e48d058b0333", "NOASSERTION"),
    ("Godson90/ai_agents", "ff6bb6b9822ff49ec62820720180c90ed9a64e3a", "NOASSERTION"),
    ("Godson90/event-planner-ai-agent", "e07bb5f1a5ae61424b03aeac2d5ba2d26e79355c", "NOASSERTION"),
    ("pverhaert/syllabot", "ffe53a4e6cfefcd052a750c4c68c9c3c096831af", "NOASSERTION"),
    ("Tanmay-Satsangi/Startup_Analyzer", "06b819ec6b81675eee00da13705907a3143c4ffc", "NOASSERTION"),
    ("VSigii/TaskSort", "e38eeca5072512fc2c1130ac21db379eb02e7fbb", "NOASSERTION"),
    ("Vidoosh/bloodhound", "e2f3b562a634f7d4f28d022c4059096b761b67b9", "MIT"),
    ("maksymvereshko/AI-Agent-System", "0f3b4dea63579b1df5ce1472d89040ecf2dc2516", "NOASSERTION"),
)

READ_TOOLS = {
    "FileReadTool",
    "DirectoryReadTool",
    "CSVSearchTool",
    "DirectorySearchTool",
    "JSONSearchTool",
    "PDFSearchTool",
    "TXTSearchTool",
    "DOCXSearchTool",
    "XMLSearchTool",
    "MDXSearchTool",
    "RagTool",
    "FixedRAGSearchTool",
}
WRITE_TOOLS = {"FileWriterTool"}
EXECUTE_TOOLS = {"CodeInterpreterTool"}
NETWORK_TOOLS = {
    "SerperDevTool",
    "ScrapeWebsiteTool",
    "WebsiteSearchTool",
    "FirecrawlScrapeWebsiteTool",
    "FirecrawlSearchTool",
    "SeleniumScrapingTool",
    "YoutubeChannelSearchTool",
    "YoutubeVideoSearchTool",
    "GithubSearchTool",
    "EXASearchTool",
    "BrowserbaseLoadTool",
    "SerpApiSearchTool",
}
KNOWN_TOOLS = READ_TOOLS | WRITE_TOOLS | EXECUTE_TOOLS | NETWORK_TOOLS
SNAKE_TOOL_ALIASES = {
    "serper_dev_tool": "SerperDevTool",
    "scrape_website_tool": "ScrapeWebsiteTool",
    "website_search_tool": "WebsiteSearchTool",
    "file_read_tool": "FileReadTool",
    "file_writer_tool": "FileWriterTool",
    "directory_read_tool": "DirectoryReadTool",
    "csv_search_tool": "CSVSearchTool",
    "code_interpreter_tool": "CodeInterpreterTool",
}
SKIP_DIRS = {".git", ".venv", "venv", "env", "__pycache__", "node_modules", "site-packages"}
EXPECTED_KEYS = {"instance_id", "source", "provenance", "task_or_role_spec", "declared_capabilities"}
PERMISSION_LEVELS = {"read", "write", "execute", "network", "admin", "unknown"}


def run_git(args: list[str], cwd: Path | None = None, timeout: int = 180) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
    )
    return proc.stdout.strip()


def checkout_repo(repo: str, commit: str, dest: Path) -> str:
    run_git(["init", str(dest)], timeout=60)
    run_git(["remote", "add", "origin", f"https://github.com/{repo}.git"], cwd=dest, timeout=60)
    run_git(["fetch", "--depth", "1", "origin", commit], cwd=dest, timeout=240)
    run_git(["checkout", "--detach", "FETCH_HEAD"], cwd=dest, timeout=60)
    return run_git(["rev-parse", "HEAD"], cwd=dest, timeout=30)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().replace("\u202f", " ")


def camel_from_snake(text: str) -> str:
    return "".join(part.capitalize() for part in text.split("_"))


def normalize_tool_name(value: Any) -> str:
    text = str(value).strip().strip("\"'")
    if not text:
        return ""
    lower = text.lower()
    if lower in SNAKE_TOOL_ALIASES:
        return SNAKE_TOOL_ALIASES[lower]

    text = text.split(".")[-1]
    text = re.sub(r"\(.*\)$", "", text).strip()
    lower = text.lower()
    if lower in SNAKE_TOOL_ALIASES:
        return SNAKE_TOOL_ALIASES[lower]

    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*Tool)\b", text)
    if match:
        return match.group(1)
    if lower.endswith("_tool"):
        candidate = camel_from_snake(lower)
        if candidate in KNOWN_TOOLS:
            return candidate
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\b", text)
    return match.group(1) if match else text


def ordered_unique(names: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in names:
        name = normalize_tool_name(raw)
        if name and name not in seen:
            out.append(name)
            seen.add(name)
    return out


def yaml_tool_names(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        out: list[Any] = []
        for item in value:
            out.extend(yaml_tool_names(item))
        return out
    if isinstance(value, dict):
        for key in ("name", "tool", "class", "type"):
            if key in value:
                return yaml_tool_names(value[key])
        return [str(key) for key in value]
    return [str(value)]


def expr_name(expr: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(expr, ast.Name):
        return env.get(expr.id, expr.id)
    if isinstance(expr, ast.Attribute):
        if isinstance(expr.value, ast.Name) and expr.value.id == "self" and expr.attr in env:
            return env[expr.attr]
        return expr.attr
    if isinstance(expr, ast.Call):
        return expr_name(expr.func, env)
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    return None


def expr_tools(expr: ast.AST | None, env: dict[str, Any]) -> list[Any]:
    if expr is None:
        return []
    if isinstance(expr, (ast.List, ast.Tuple, ast.Set)):
        out: list[Any] = []
        for elt in expr.elts:
            out.extend(expr_tools(elt, env))
        return out
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        return expr_tools(expr.left, env) + expr_tools(expr.right, env)
    if isinstance(expr, ast.Name) and expr.id in env:
        value = env[expr.id]
        return value if isinstance(value, list) else [value]
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name) and expr.value.id == "self":
        if expr.attr in env:
            value = env[expr.attr]
            return value if isinstance(value, list) else [value]
    name = expr_name(expr, env)
    return [name] if name else []


def assignment_value(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return expr_tools(node, env)
    if isinstance(node, ast.Call):
        return expr_name(node.func, env)
    if isinstance(node, ast.Name):
        return env.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return expr_name(node, env)
    return None


def parse_py_tools(path: Path) -> dict[str, list[str]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}

    env: dict[str, Any] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                env[alias.asname or alias.name] = alias.name

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = assignment_value(node.value, env)
        if value is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                env[target.id] = value
            elif isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                env[target.attr] = value

    out: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call) or expr_name(call.func, env) != "Agent":
                continue
            tools_expr: ast.AST | None = None
            config_key: str | None = None
            for keyword in call.keywords:
                if keyword.arg == "tools":
                    tools_expr = keyword.value
                elif keyword.arg == "config" and isinstance(keyword.value, ast.Subscript):
                    slice_node = keyword.value.slice
                    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
                        config_key = slice_node.value
            names = ordered_unique(expr_tools(tools_expr, env))
            if not names:
                continue
            out[node.name] = ordered_unique(out.get(node.name, []) + names)
            if config_key:
                out[config_key] = ordered_unique(out.get(config_key, []) + names)
    return out


def find_agent_yamls(repo_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in repo_dir.rglob("agents.yaml")
        if "config" in path.relative_to(repo_dir).parts
        and not any(part in SKIP_DIRS for part in path.relative_to(repo_dir).parts)
    )


def find_python_tool_maps(repo_dir: Path) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for path in repo_dir.rglob("*.py"):
        rel_parts = path.relative_to(repo_dir).parts
        if any(part in SKIP_DIRS for part in rel_parts):
            continue
        name = path.name.lower()
        if not (name == "crew.py" or name.endswith("_crew.py") or name in {"agents.py", "tools.py"}):
            continue
        for key, names in parse_py_tools(path).items():
            merged[key] = ordered_unique(merged.get(key, []) + names)
    return merged


def capability(name: str) -> dict[str, str]:
    name = normalize_tool_name(name)
    if name in READ_TOOLS:
        permission = "read"
    elif name in WRITE_TOOLS:
        permission = "write"
    elif name in EXECUTE_TOOLS:
        permission = "execute"
    elif name in NETWORK_TOOLS:
        permission = "network"
    else:
        permission = "unknown"
    return {
        "name": name,
        "type": f"crewai_tools.{name}" if name in KNOWN_TOOLS else "custom",
        "permission_level": permission,
    }


def role_spec(agent: dict[str, Any]) -> str:
    role = clean_text(agent.get("role")) or clean_text(agent.get("name"))
    goal = clean_text(agent.get("goal"))
    backstory = clean_text(agent.get("backstory"))
    parts: list[str] = []
    if role:
        parts.append(f"Role: {role}")
    if goal:
        parts.append(f"Goal: {goal}")
    if backstory and len(backstory) <= 500 and len(backstory.split()) <= 90:
        parts.append(f"Backstory: {backstory}")
    return "\n".join(parts)


def parse_repo(repo_dir: Path, repo: str, commit: str, license_id: str) -> tuple[list[dict[str, Any]], int]:
    tool_map = find_python_tool_maps(repo_dir)
    rows: list[dict[str, Any]] = []
    raw_agents = 0
    for yaml_path in find_agent_yamls(repo_dir):
        try:
            data = yaml.safe_load(yaml_path.read_text(encoding="utf-8", errors="ignore"))
        except Exception as exc:
            print(f"skip_yaml={repo}:{yaml_path.name}:{exc}", file=sys.stderr)
            continue
        if not isinstance(data, dict):
            continue
        rel_path = yaml_path.relative_to(repo_dir).as_posix()
        for agent_key, agent in data.items():
            if not isinstance(agent, dict):
                continue
            spec = role_spec(agent)
            if not spec:
                continue
            raw_agents += 1
            tools = ordered_unique(yaml_tool_names(agent.get("tools"))) or tool_map.get(str(agent_key), [])
            if not tools:
                continue
            rows.append(
                {
                    "instance_id": "",
                    "source": "crewai",
                    "provenance": {
                        "repo": repo,
                        "commit": commit,
                        "path": rel_path,
                        "license": license_id,
                    },
                    "task_or_role_spec": spec,
                    "declared_capabilities": [capability(name) for name in tools],
                }
            )
    return rows, raw_agents


def role_key(row: dict[str, Any]) -> str:
    spec = row["task_or_role_spec"]
    role = spec.split("\n", 1)[0]
    if role.startswith("Role: "):
        role = role[len("Role: ") :]
    return re.sub(r"\s+", " ", role).strip().lower()


def dedup_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for row in rows:
        tools = tuple(sorted(cap["name"] for cap in row["declared_capabilities"]))
        key = (role_key(row), tools)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
        if len(out) >= limit:
            break
    for index, row in enumerate(out, start=1):
        row["instance_id"] = f"crewai-{index:04d}"
    return out


def validate_rows(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        assert set(row) == EXPECTED_KEYS, row.get("instance_id")
        assert row["source"] == "crewai"
        for key in ("repo", "commit", "path", "license"):
            assert key in row["provenance"], key
        assert row["declared_capabilities"], row["instance_id"]
        for cap in row["declared_capabilities"]:
            assert set(cap) == {"name", "type", "permission_level"}, cap
            assert cap["permission_level"] in PERMISSION_LEVELS, cap


def harvest(max_rows: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    raw_agents = 0
    skipped = 0
    with tempfile.TemporaryDirectory(prefix="crewai-src-") as temp:
        base = Path(temp)
        for repo, commit, license_id in SOURCE_REPOS:
            dest = base / repo.replace("/", "__")
            try:
                actual_commit = checkout_repo(repo, commit, dest)
                rows, raw = parse_repo(dest, repo, actual_commit, license_id)
            except Exception as exc:
                skipped += 1
                print(f"skip_repo={repo}:{exc}", file=sys.stderr)
                continue
            raw_agents += raw
            all_rows.extend(rows)
            print(f"repo={repo} raw_agents={raw} tool_rows={len(rows)}", file=sys.stderr)

    deduped = dedup_rows(all_rows, max_rows)
    validate_rows(deduped)
    tool_counts = [len(row["declared_capabilities"]) for row in deduped]
    summary = {
        "source_repos": len(SOURCE_REPOS),
        "skipped_repos": skipped,
        "raw_agents": raw_agents,
        "raw_tool_rows": len(all_rows),
        "written": len(deduped),
        "dedup_ratio_raw_to_written": round(len(all_rows) / len(deduped), 3) if deduped else 0.0,
        "mean_tools_per_config": round(statistics.mean(tool_counts), 3) if tool_counts else 0.0,
        "license_distribution": dict(sorted(Counter(row["provenance"]["license"] for row in deduped).items())),
    }
    return deduped, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT_PATH)
    parser.add_argument("--max-rows", type=int, default=300)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, summary = harvest(args.max_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    print(f"wrote={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
