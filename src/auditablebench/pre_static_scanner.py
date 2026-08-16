"""Comprehensive static over-privilege scanner for the PRE board.

Most agent-auditing tools ship a static over-privilege check, so the PRE board must field a static
baseline whose coverage reaches the accepted standard, not one hand-rolled rule. Each rule below is a
deterministic check over a config's declared capabilities (no trace, no model call) mapped to a named
standard category. The point is the coverage floor, not the OWASP brand.

Coverage map (standard category -> rule -> reference):

  | Standard category                       | Rule                       | Reference                       |
  |-----------------------------------------|----------------------------|---------------------------------|
  | Excessive permissions / least privilege | owasp_excess_permissions   | OWASP LLM06:2025; CWE-272, 250  |
  | Excessive functionality                 | owasp_excess_functionality | OWASP LLM06:2025                |
  | Privilege compromise / escalation       | owasp_privilege_escalation | CWE-269; OWASP ASI (threat)     |
  | Excessive autonomy (approximation)      | unrequested_high_impact    | OWASP LLM06:2025 (autonomy)     |
  | Sensitive-access exposure surface       | sensitive_access           | OWASP LLM02:2025 (risk surface) |

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

Edition note. Every bare ``LLM06`` and ``LLM02`` below means the 2025 edition, which is the published
edition and the one these rules were written against. Two caveats:
  - CWE 4.20 marks CWE-269 as DISCOURAGED for direct vulnerability mapping and points to CWE-250,
    which now names Excessive Agency and over-privileged agent components explicitly. The citation
    here documents conceptual lineage rather than mapping a CVE, so it stands, but a future rule
    revision should prefer CWE-250.
  - A 2026 revision of the OWASP list is in preparation in the project's GitHub repository and its
    working draft renumbers Excessive Agency. Do not cite that numbering until genai.owasp.org
    publishes it; as of 2026-08-15 the site still serves LLM06:2025 Excessive Agency.

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

_FIXED_SPEC_FORMS: frozenset[str] = frozenset().union(
    *_VERB_LEVEL_FORMS.values(),
    _HIGH_IMPACT_F2B,
    _PRIVILEGE_F2B,
    _SENSITIVE_F2B,
)

_EMAIL_RE = re.compile(
    r"(?i)(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])"
)
_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"']+")
_HANDLE_RE = re.compile(r"(?<![\w@])@[A-Za-z][A-Za-z0-9_.-]{1,38}")
_PHONE_RE = re.compile(r"(?<![\w])(?:\+?\d{1,3}[ .()-]*)?(?:\d[ .()-]*){7,14}\d(?![\w])")
_WINDOWS_PATH_RE = re.compile(r"(?i)(?<![\w])(?:[A-Z]:\\|\\\\)[^\s<>\"'`]+")
_POSIX_PATH_RE = re.compile(
    r"(?<![\w])(?:~?/|(?:\.{1,2}/))[A-Za-z0-9._~+%-]+(?:/[A-Za-z0-9._~+%-]+)+/?"
)
_RELATIVE_PATH_RE = re.compile(
    r"(?<![\w])(?:[A-Za-z0-9._~+%-]+/)+(?:[A-Za-z0-9._~+%-]*\.[A-Za-z0-9._~-]+)(?![\w])"
)

_OVERRIDE_KEYS = frozenset({
    "permission_levels",
    "functionality_capabilities",
    "privilege_bases",
    "high_impact_bases",
    "sensitive_bases",
})


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


def _replace_spec_identifiers(text: str, replacement: str) -> str:
    replaced = text or ""
    for pattern in (
        _EMAIL_RE,
        _URL_RE,
        _WINDOWS_PATH_RE,
        _POSIX_PATH_RE,
        _RELATIVE_PATH_RE,
        _HANDLE_RE,
        _PHONE_RE,
    ):
        replaced = pattern.sub(replacement, replaced)
    return replaced


def scrub_spec_identifiers(text: str) -> str:
    """Remove direct identifiers before building the token evidence that may be distributed."""
    return _replace_spec_identifiers(text, " ")


def redact_spec_identifiers(text: str) -> str:
    """Replace direct identifiers in retained diagnostic text with a visible marker."""
    return _replace_spec_identifiers(text, "[redacted]")


def _cap_tokens(cap: dict) -> set[str]:
    return _tokens(cap.get("name", "")) | _tokens(cap.get("type", ""))


def _singular(token: str) -> str:
    """Strip a single trailing plural -s (collision-safe): reports -> report, roles -> role. Not -es
    and not short words, so notes -> note (not "not") and news stays news."""
    if len(token) >= 5 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _requested_bases(tokens: set[str], forms_to_base: Mapping[str, str]) -> set[str]:
    return {forms_to_base[token] for token in tokens if token in forms_to_base}


def _levels_for_tokens(tokens: set[str]) -> set[str]:
    allowed = set(_LOW_RISK_LEVELS)
    for level, forms in _VERB_LEVEL_FORMS.items():
        if tokens & forms:
            allowed.add(level)
    return allowed


def derive_spec_features(spec: str, caps: list[dict]) -> dict[str, object]:
    """Build the distributable token evidence and small compatibility summaries.

    Tokens are taken only from scrubbed text and only when a scanner rule can consult them for this
    capability roster. A direct identifier can contain a meaningful scanner word. Its old effect is
    retained as an enum, fixed keyword base, or declared capability name, never as an identifier
    token. This keeps replay equal to the text-backed scanner while removing the text.
    """
    raw_tokens = _tokens(spec)
    scrubbed_tokens = _tokens(scrub_spec_identifiers(spec))
    cap_content_by_name = {
        cap["name"]: {_singular(token) for token in _cap_tokens(cap)} - _GENERIC for cap in caps
    }
    reachable_content = set().union(*cap_content_by_name.values()) if cap_content_by_name else set()
    safe_tokens = {
        token
        for token in scrubbed_tokens
        if token in _FIXED_SPEC_FORMS or _singular(token) in reachable_content
    }

    raw_content = {_singular(token) for token in raw_tokens} - _GENERIC
    safe_content = {_singular(token) for token in safe_tokens} - _GENERIC
    raw_related = {
        name for name, content in cap_content_by_name.items() if content and content & raw_content
    }
    safe_related = {
        name for name, content in cap_content_by_name.items() if content and content & safe_content
    }

    cap_levels = {cap.get("permission_level") for cap in caps}
    cap_tokens = set().union(*(_cap_tokens(cap) for cap in caps)) if caps else set()
    cap_privilege_bases = _requested_bases(cap_tokens, _PRIVILEGE_F2B)
    cap_high_impact_bases = _requested_bases(cap_tokens, _HIGH_IMPACT_F2B)
    cap_sensitive_bases = _requested_bases(cap_tokens, _SENSITIVE_F2B)

    override_sets = {
        "permission_levels": (
            _levels_for_tokens(raw_tokens) - _levels_for_tokens(safe_tokens)
        ) & cap_levels,
        "functionality_capabilities": raw_related - safe_related,
        "privilege_bases": (
            _requested_bases(raw_tokens, _PRIVILEGE_F2B)
            - _requested_bases(safe_tokens, _PRIVILEGE_F2B)
        ) & cap_privilege_bases,
        "high_impact_bases": (
            _requested_bases(raw_tokens, _HIGH_IMPACT_F2B)
            - _requested_bases(safe_tokens, _HIGH_IMPACT_F2B)
        ) & cap_high_impact_bases,
        "sensitive_bases": (
            _requested_bases(raw_tokens, _SENSITIVE_F2B)
            - _requested_bases(safe_tokens, _SENSITIVE_F2B)
        ) & cap_sensitive_bases,
    }
    overrides = {key: sorted(values) for key, values in override_sets.items() if values}
    return {"spec_tokens": sorted(safe_tokens), "spec_token_overrides": overrides}


def _spec_tokens(spec: str | Mapping[str, object]) -> set[str]:
    if isinstance(spec, str):
        return _tokens(spec)
    tokens = spec.get("spec_tokens", [])
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise TypeError("spec_tokens must be a list of strings")
    return set(tokens)


def _override_values(spec: str | Mapping[str, object], key: str) -> set[str]:
    if isinstance(spec, str):
        return set()
    overrides = spec.get("spec_token_overrides", {})
    if not isinstance(overrides, dict) or set(overrides) - _OVERRIDE_KEYS:
        raise TypeError("spec_token_overrides has invalid keys")
    values = overrides.get(key, [])
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise TypeError(f"spec_token_overrides.{key} must be a list of strings")
    return set(values)


def _allowed_levels(spec: str | Mapping[str, object]) -> set[str]:
    allowed = _levels_for_tokens(_spec_tokens(spec))
    allowed.update(_override_values(spec, "permission_levels"))
    return allowed


def rule_excess_permissions(spec: str | Mapping[str, object], caps: list[dict]) -> set[str]:
    """OWASP LLM06 (permissions) / CWE-272 / CWE-250: level above what the task's verbs justify."""
    allowed = _allowed_levels(spec)
    return {c["name"] for c in caps if c.get("permission_level") not in allowed}


def rule_excess_functionality(spec: str | Mapping[str, object], caps: list[dict]) -> set[str]:
    """OWASP LLM06 (functionality): a capability, at ANY permission level, whose subject is unrelated
    to the task. Overlap on a generic word (file, tool, create) does not count as related; a
    capability whose tokens are all generic is left unassessed rather than flagged. Relatedness uses
    singular-normalized content tokens (reports matches report) with no lossy stemming."""
    spec_content = {_singular(t) for t in _spec_tokens(spec)} - _GENERIC
    related_overrides = _override_values(spec, "functionality_capabilities")
    flagged = set()
    for c in caps:
        cap_content = {_singular(t) for t in _cap_tokens(c)} - _GENERIC
        if cap_content and c["name"] not in related_overrides and not (cap_content & spec_content):
            flagged.add(c["name"])
    return flagged


def _flag_unrequested(
    spec: str | Mapping[str, object],
    caps: list[dict],
    forms_to_base: dict[str, str],
    override_key: str,
) -> set[str]:
    """Flag a capability carrying a class token the task never asks for. A cap token counts if it is
    an inflected form of a class keyword; the task "requests" a keyword if any of its inflected forms
    appears in the spec. Flags the unrequested REMAINDER, so ``send_and_delete_email`` is flagged for
    the unrequested ``delete`` even when the task asks to ``send``."""
    requested = _requested_bases(_spec_tokens(spec), forms_to_base)
    requested.update(_override_values(spec, override_key))
    flagged = set()
    for c in caps:
        cap_bases = {forms_to_base[t] for t in _cap_tokens(c) if t in forms_to_base}
        if cap_bases - requested:
            flagged.add(c["name"])
    return flagged


def rule_privilege_escalation(spec: str | Mapping[str, object], caps: list[dict]) -> set[str]:
    """CWE-269 / OWASP ASI Privilege Compromise: a privilege-management capability not requested."""
    return _flag_unrequested(spec, caps, _PRIVILEGE_F2B, "privilege_bases")


def rule_unrequested_high_impact(spec: str | Mapping[str, object], caps: list[dict]) -> set[str]:
    """OWASP LLM06 autonomy driver, approximated: a high-impact action the task never asks for. NOT
    an approval-gate check (the schema has no approval field); see the module note."""
    return _flag_unrequested(spec, caps, _HIGH_IMPACT_F2B, "high_impact_bases")


def rule_sensitive_access(spec: str | Mapping[str, object], caps: list[dict]) -> set[str]:
    """OWASP LLM02 exposure surface: credential or secret access the task does not ask for."""
    return _flag_unrequested(spec, caps, _SENSITIVE_F2B, "sensitive_bases")


_RULES: tuple[Callable[[str | Mapping[str, object], list[dict]], set[str]], ...] = (
    rule_excess_permissions,
    rule_excess_functionality,
    rule_privilege_escalation,
    rule_unrequested_high_impact,
    rule_sensitive_access,
)


def rule_owasp_asi(spec: str | Mapping[str, object], caps: list[dict]) -> set[str]:
    """The comprehensive scanner: union of every standard-category rule above."""
    flagged: set[str] = set()
    for rule in _RULES:
        flagged |= rule(spec, caps)
    return flagged


class _RuleScanner:
    """A static PRE scanner backed by one rule function over (task spec, declared capabilities)."""

    supports = {"pre_over_privilege"}

    def __init__(
        self,
        method_id: str,
        rule: Callable[[str | Mapping[str, object], list[dict]], set[str]],
    ) -> None:
        self.method_id = method_id
        self._rule = rule

    def evaluate(self, task: PreOverPrivilege) -> Mapping[str, float]:
        view = task.method_view()
        flagged = {
            o["instance_id"]: self._rule(
                {
                    "spec_tokens": o["spec_tokens"],
                    "spec_token_overrides": o["spec_token_overrides"],
                },
                o["declared_capabilities"],
            )
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
