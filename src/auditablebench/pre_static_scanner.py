"""Comprehensive static over-privilege scanner for the PRE board.

Most agent-auditing tools ship a static over-privilege check, so the PRE board must field a static
baseline whose coverage reaches the accepted standard, not one hand-rolled rule. Each rule below is a
deterministic check over a config's declared capabilities (no trace, no model call) mapped to a named
standard category. The point is the coverage floor, not the OWASP brand.

Coverage map (standard category -> rule -> reference):

  | Standard category                       | Rule                       | Reference                       |
  |-----------------------------------------|----------------------------|---------------------------------|
  | Excessive permissions / least privilege | owasp_excess_permissions   | OWASP LLM06; CWE-272; CWE-250   |
  | Excessive functionality                 | owasp_excess_functionality | OWASP LLM06                     |
  | Privilege compromise / escalation       | owasp_privilege_escalation | CWE-269; OWASP ASI (threat)     |
  | Excessive autonomy (approximation)      | unrequested_high_impact    | OWASP LLM06 (autonomy driver)   |
  | Sensitive-access exposure surface       | sensitive_access           | OWASP LLM02 (risk surface)      |

``owasp_asi_combined`` is the union of all five, the comprehensive scanner.

What a single-config static rule CANNOT decide, disclosed rather than overclaimed:

  - Full excessive-autonomy: OWASP's driver is a high-impact action allowed WITHOUT independent
    verification or approval. The declared-capability schema carries no approval-gate field, so
    ``unrequested_high_impact`` is an approximation (a high-impact action the task never asks for),
    not an approval-gate check. A true check needs a ``requires_independent_approval``-style field on
    the harvested capability.
  - Full ASI Tool Misuse: misuse turns on declared controls (allowed operations, resource scope,
    command / URL / destination allowlists, argument schema) the current schema does not express.
    ``sensitive_access`` is a narrower LLM02 exposure heuristic, not Tool Misuse coverage.
  - A deprecated or duplicate extension left available needs deployment history, not one config.

References:
  OWASP Top 10 for LLM Applications 2025, LLM06:2025 Excessive Agency, drivers: excessive
    functionality, excessive permissions, excessive autonomy
    (https://genai.owasp.org/llmrisk/llm062025-excessive-agency/).
  OWASP Agentic Security Initiative, "Agentic AI - Threats and Mitigations": Privilege Compromise
    and Tool Misuse are threat classes; Least-Privilege and Least-Agency are its mitigation
    principles (https://genai.owasp.org/resource/agentic-ai-threats-and-mitigations/).
  CWE-250 Execution with Unnecessary Privileges; CWE-269 Improper Privilege Management;
  CWE-272 Least Privilege Violation (https://cwe.mitre.org/).

Inflection is handled by expanding each KNOWN keyword forward to its inflected forms (``create`` ->
``creating``, ``created``, ``creates``) and matching spec tokens against those forms by exact
membership. This is collision-safe: an unrelated word never merges into a keyword (``note`` is not a
form of ``not``, ``installing`` matches ``install`` without a lossy stem). The rules stay keyword-
based and therefore language-brittle: a task spec in a language the keyword lists do not cover falls
back to the least-privilege floor and over-flags, a property reported per source, not hidden, and
part of why the board also carries a held-out LLM judge.
"""
from __future__ import annotations

import re
from typing import Callable, Mapping

from .pre import PreOverPrivilege, pre_score

_LOW_RISK_LEVELS = {"read", "unknown"}

# Verb families -> the permission level a task implies when a matching verb appears in its spec.
# "read" and "unknown" are always allowed as levels (the least-privilege floor for the permissions
# rule); unnecessary read/unknown CAPABILITIES are still caught by the functionality rule.
_VERB_LEVELS: dict[str, tuple[str, ...]] = {
    "network": (
        "fetch", "http", "https", "url", "download", "request", "scrape", "browse", "crawl",
        "search", "query",
    ),
    "write": (
        "write", "save", "edit", "create", "update", "delete", "modify", "store", "insert",
        "append", "remove", "generate", "draft", "compose", "upload", "export", "publish",
    ),
    "execute": (
        "run", "execute", "shell", "compile", "build", "deploy", "install", "calculate",
        "compute", "simulate",
    ),
    "admin": (
        "administrate", "configure", "provision", "grant", "revoke", "sudo",
    ),
}

# High-impact / irreversible ACTION tokens (OWASP LLM06 autonomy driver, approximated): destructive,
# execution, and outbound actions. See the module note: without an approval-gate field this is an
# "unrequested high-impact action" heuristic, not a true autonomy check.
_HIGH_IMPACT_ACTION: frozenset[str] = frozenset({
    "delete", "drop", "remove", "destroy", "truncate", "purge", "wipe", "terminate",
    "shutdown", "reboot", "rm", "exec", "execute", "shell", "bash", "deploy", "publish",
    "release", "install", "migrate", "transfer", "pay", "payment", "wire", "withdraw",
    "send", "email", "post", "tweet", "message", "upload",
})

# Privilege-management tokens (CWE-269; OWASP ASI Privilege Compromise threat): capabilities that
# grant, manage, or escalate privilege.
_PRIVILEGE_MGMT: frozenset[str] = frozenset({
    "grant", "revoke", "admin", "administrate", "role", "policy", "iam", "permission",
    "provision", "root", "sudo", "escalate", "privilege",
})

# Sensitive-resource tokens (OWASP LLM02 exposure surface): credential and secret access.
_SENSITIVE: frozenset[str] = frozenset({
    "credential", "secret", "password", "passwd", "apikey",
    "token", "ssh", "privatekey", "keypair", "vault", "keychain", "key",
})

# Generic tokens that do NOT establish that a capability is related to a task's subject. Overlap on
# one of these must not clear the functionality rule (create, file, tool, data, and so on).
_GENERIC: frozenset[str] = frozenset({
    "tool", "tools", "api", "apis", "service", "services", "function", "functions", "agent",
    "helper", "util", "utils", "utility", "data", "info", "information", "get", "set", "use",
    "run", "the", "and", "for", "with", "create", "read", "write", "list", "file", "files",
    "manager", "client", "handler", "system", "task", "action", "execute", "process", "request",
    "call", "access", "value", "input", "output", "result", "item", "object", "content",
})

# Short security-relevant tokens kept even under the length filter.
_KEEP_SHORT: frozenset[str] = frozenset({"rm", "db", "fs", "s3", "ls", "cp", "mv", "iam", "ssh"})

# Known irregular inflections, keyword base -> extra forms, added forward (never merged backward).
_IRREGULAR_FORMS: dict[str, set[str]] = {
    "run": {"ran"}, "write": {"wrote", "written"}, "send": {"sent"}, "build": {"built"},
    "pay": {"paid"}, "withdraw": {"withdrew", "withdrawn"},
}

# Keywords that take verb inflection (-ed / -ing). Everything else in a class is treated as a noun
# and takes only a plural, so a noun token like ``secret`` or ``vault`` never generates ``secretes``
# / ``vaulting`` and cannot be cleared by an unrelated gland or gymnastics spec.
_VERBS: frozenset[str] = frozenset(
    {v for kws in _VERB_LEVELS.values() for v in kws}
    | {
        "delete", "drop", "remove", "destroy", "truncate", "purge", "wipe", "terminate",
        "shutdown", "reboot", "exec", "deploy", "publish", "release", "migrate", "transfer",
        "pay", "wire", "withdraw", "send", "post", "tweet", "upload", "email", "message",
        "grant", "revoke", "administrate", "provision", "escalate",
    }
)
_SIBILANTS = ("s", "x", "z", "ch", "sh")


def _is_cvc(b: str) -> bool:
    """True for a consonant-vowel-consonant final (run, transfer): the final consonant doubles
    before -ed / -ing. Excludes final w/x/y and roots already ending in a double (install, shell)."""
    return len(b) >= 3 and b[-1] not in "aeiouwxy" and b[-2] in "aeiou" and b[-3] not in "aeiou"


def _plural(b: str) -> str:
    if b.endswith(_SIBILANTS):
        return b + "es"                       # search -> searches (not run -> runes)
    if b.endswith("y") and b[-2] not in "aeiou":
        return b[:-1] + "ies"                 # query -> queries (consonant + y)
    return b + "s"                            # run -> runs; deploy -> deploys (vowel + y)


def _inflect(base: str) -> set[str]:
    """Forward inflected forms of one keyword. Verbs get the plural plus -ed / -ing (drop-e, CVC
    doubling, consonant-y -> -ied / -ying) and any known irregulars; nouns get only the plural. Only
    KNOWN keywords are expanded, so matching a spec token against these forms by exact membership
    cannot merge an unrelated word into a keyword."""
    b = base.lower()
    if len(b) < 3:
        return {b}                            # too short to inflect (rm, s3, iam, ssh): literal
    forms = {b, _plural(b)}
    if b in _VERBS:
        if b.endswith("e"):
            forms.update({b + "d", b[:-1] + "ing"})            # create -> created, creating
        elif b.endswith("y") and b[-2] not in "aeiou":
            forms.update({b[:-1] + "ied", b + "ing"})          # query -> queried, querying
        else:
            forms.update({b + "ed", b + "ing"})               # deploy -> deployed, deploying
            if _is_cvc(b):
                forms.update({b + b[-1] + "ed", b + b[-1] + "ing"})  # transfer -> transferred
        forms |= _IRREGULAR_FORMS.get(b, set())
    return forms


def _forms_to_base(words) -> dict[str, str]:
    """Reverse map: every inflected form -> its base keyword. Bases are visited in a fixed (length,
    then lexicographic) order and the first base wins a shared form, so the map is deterministic
    across hash seeds."""
    out: dict[str, str] = {}
    for base in sorted(words, key=lambda w: (len(w), w)):
        for form in _inflect(base):
            out.setdefault(form, base)
    return out


_VERB_LEVEL_FORMS: dict[str, frozenset[str]] = {
    level: frozenset().union(*(_inflect(k) for k in kws)) for level, kws in _VERB_LEVELS.items()
}
_HIGH_IMPACT_F2B = _forms_to_base(_HIGH_IMPACT_ACTION)
_PRIVILEGE_F2B = _forms_to_base(_PRIVILEGE_MGMT)
_SENSITIVE_F2B = _forms_to_base(_SENSITIVE)


def _tokens(text: str) -> set[str]:
    """Lowercased token set. Splits snake/kebab and camelCase INCLUDING acronym boundaries
    (``IAMRole`` -> ``iam``, ``role``; ``SSHPrivateKey`` -> ``ssh``, ``private``, ``key``). Keeps
    tokens of length 3+, plus a small set of short security tokens (``rm``, ``s3``, ...)."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text or "")
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
    out = set()
    for part in re.split(r"[^A-Za-z0-9]+", spaced):
        low = part.lower()
        if len(low) >= 3 or low in _KEEP_SHORT:
            out.add(low)
    return out


def _cap_tokens(cap: dict) -> set[str]:
    return _tokens(cap.get("name", "")) | _tokens(cap.get("type", ""))


def _singular(token: str) -> str:
    """Strip a single trailing plural -s (collision-safe): reports -> report, roles -> role. Not -es
    and not short words, so notes -> note (not "not") and news stays news."""
    if len(token) >= 5 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _allowed_levels(spec: str) -> set[str]:
    spec_tokens = _tokens(spec)
    allowed = set(_LOW_RISK_LEVELS)
    for level, forms in _VERB_LEVEL_FORMS.items():
        if spec_tokens & forms:
            allowed.add(level)
    return allowed


def rule_excess_permissions(spec: str, caps: list[dict]) -> set[str]:
    """OWASP LLM06 (permissions) / CWE-272 / CWE-250: level above what the task's verbs justify."""
    allowed = _allowed_levels(spec)
    return {c["name"] for c in caps if c.get("permission_level") not in allowed}


def rule_excess_functionality(spec: str, caps: list[dict]) -> set[str]:
    """OWASP LLM06 (functionality): a capability, at ANY permission level, whose subject is unrelated
    to the task. Overlap on a generic word (file, tool, create) does not count as related; a
    capability whose tokens are all generic is left unassessed rather than flagged. Relatedness uses
    singular-normalized content tokens (reports matches report) with no lossy stemming."""
    spec_content = {_singular(t) for t in _tokens(spec)} - _GENERIC
    flagged = set()
    for c in caps:
        cap_content = {_singular(t) for t in _cap_tokens(c)} - _GENERIC
        if cap_content and not (cap_content & spec_content):
            flagged.add(c["name"])
    return flagged


def _flag_unrequested(spec: str, caps: list[dict], forms_to_base: dict[str, str]) -> set[str]:
    """Flag a capability carrying a class token the task never asks for. A cap token counts if it is
    an inflected form of a class keyword; the task "requests" a keyword if any of its inflected forms
    appears in the spec. Flags the unrequested REMAINDER, so ``send_and_delete_email`` is flagged for
    the unrequested ``delete`` even when the task asks to ``send``."""
    spec_tokens = _tokens(spec)
    requested = {forms_to_base[t] for t in spec_tokens if t in forms_to_base}
    flagged = set()
    for c in caps:
        cap_bases = {forms_to_base[t] for t in _cap_tokens(c) if t in forms_to_base}
        if cap_bases - requested:
            flagged.add(c["name"])
    return flagged


def rule_privilege_escalation(spec: str, caps: list[dict]) -> set[str]:
    """CWE-269 / OWASP ASI Privilege Compromise: a privilege-management capability not requested."""
    return _flag_unrequested(spec, caps, _PRIVILEGE_F2B)


def rule_unrequested_high_impact(spec: str, caps: list[dict]) -> set[str]:
    """OWASP LLM06 autonomy driver, approximated: a high-impact action the task never asks for. NOT
    an approval-gate check (the schema has no approval field); see the module note."""
    return _flag_unrequested(spec, caps, _HIGH_IMPACT_F2B)


def rule_sensitive_access(spec: str, caps: list[dict]) -> set[str]:
    """OWASP LLM02 exposure surface: credential or secret access the task does not ask for."""
    return _flag_unrequested(spec, caps, _SENSITIVE_F2B)


_RULES: tuple[Callable[[str, list[dict]], set[str]], ...] = (
    rule_excess_permissions,
    rule_excess_functionality,
    rule_privilege_escalation,
    rule_unrequested_high_impact,
    rule_sensitive_access,
)


def rule_owasp_asi(spec: str, caps: list[dict]) -> set[str]:
    """The comprehensive scanner: union of every standard-category rule above."""
    flagged: set[str] = set()
    for rule in _RULES:
        flagged |= rule(spec, caps)
    return flagged


class _RuleScanner:
    """A static PRE scanner backed by one rule function over (task spec, declared capabilities)."""

    supports = {"pre_over_privilege"}

    def __init__(self, method_id: str, rule: Callable[[str, list[dict]], set[str]]) -> None:
        self.method_id = method_id
        self._rule = rule

    def evaluate(self, task: PreOverPrivilege) -> Mapping[str, float]:
        view = task.method_view()
        flagged = {
            o["instance_id"]: self._rule(o["task_or_role_spec"], o["declared_capabilities"])
            for o in view
        }
        return pre_score(flagged, task.instances)


def owasp_scanner_methods() -> list:
    """The standard-coverage static-scanner set: the five category rules plus their union."""
    return [
        _RuleScanner("owasp_excess_permissions", rule_excess_permissions),
        _RuleScanner("owasp_excess_functionality", rule_excess_functionality),
        _RuleScanner("owasp_privilege_escalation", rule_privilege_escalation),
        _RuleScanner("unrequested_high_impact", rule_unrequested_high_impact),
        _RuleScanner("sensitive_access", rule_sensitive_access),
        _RuleScanner("owasp_asi_combined", rule_owasp_asi),
    ]
