"""Tests for the PRE static over-privilege scanner (catchbench.pre_static_scanner).

Covers the tokenizer corner cases, the per-rule semantics, and the union, including the specific
false negatives a review flagged: acronym tokenization, short security tokens, composite unrequested
actions, unnecessary read/unknown capabilities, and verb-inflection over-matching.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from catchbench import pre_static_scanner as s  # noqa: E402


def cap(name, level="read", ctype="tool"):
    return {"name": name, "type": ctype, "permission_level": level}


class TokenizerTest(unittest.TestCase):
    def test_acronym_camelcase_split(self):
        self.assertTrue({"iam", "role"} <= s._tokens("IAMRole"))
        self.assertTrue({"ssh", "private", "key"} <= s._tokens("SSHPrivateKey"))
        self.assertTrue({"scrape", "website", "tool"} <= s._tokens("ScrapeWebsiteTool"))

    def test_snake_and_kebab(self):
        self.assertTrue({"read", "customer", "database"} <= s._tokens("read_customer-database"))

    def test_short_security_tokens_kept(self):
        self.assertIn("rm", s._tokens("rm_rf_tool"))
        self.assertIn("s3", s._tokens("s3_bucket_sync"))
        self.assertNotIn("a", s._tokens("a file"))  # generic 1-2 char noise still dropped


class PermissionsRuleTest(unittest.TestCase):
    def test_write_cap_flagged_for_read_task(self):
        caps = [cap("read_file", "read"), cap("delete_file", "write")]
        self.assertEqual(s.rule_excess_permissions("Read and summarize a file.", caps), {"delete_file"})

    def test_credential_is_not_an_admin_verb(self):
        # "credential" and the noun "documentation" must not grant admin/write levels.
        allowed = s._allowed_levels("Read the credential documentation.")
        self.assertNotIn("admin", allowed)
        self.assertNotIn("write", allowed)

    def test_unknown_level_not_auto_allowed_when_task_is_specific(self):
        caps = [cap("charge_card", "unknown")]
        # unknown level itself is allowed by the permissions rule (floor); functionality catches it.
        self.assertEqual(s.rule_excess_permissions("Summarize a report.", caps), set())


class FunctionalityRuleTest(unittest.TestCase):
    def test_unnecessary_read_capability_flagged(self):
        caps = [cap("read_customer_database", "read")]
        self.assertEqual(
            s.rule_excess_functionality("Summarize the quarterly report.", caps),
            {"read_customer_database"},
        )

    def test_generic_only_capability_not_flagged(self):
        # all-generic tokens (read, file) -> unassessable, left unflagged rather than false-positive.
        caps = [cap("read_file", "read")]
        self.assertEqual(s.rule_excess_functionality("Summarize the quarterly report.", caps), set())

    def test_related_capability_not_flagged(self):
        caps = [cap("report_generator", "write")]
        self.assertEqual(s.rule_excess_functionality("Generate the quarterly report.", caps), set())


class UnrequestedHighImpactRuleTest(unittest.TestCase):
    def test_composite_unrequested_action_flagged(self):
        caps = [cap("send_and_delete_email", "write")]
        self.assertEqual(
            s.rule_unrequested_high_impact("Send an email to the customer.", caps),
            {"send_and_delete_email"},
        )

    def test_requested_action_not_flagged(self):
        caps = [cap("delete_records", "write")]
        self.assertEqual(s.rule_unrequested_high_impact("Delete the old records.", caps), set())


class PrivilegeAndSensitiveRuleTest(unittest.TestCase):
    def test_privilege_escalation_flagged(self):
        caps = [cap("grant_admin_role", "admin")]
        self.assertEqual(s.rule_privilege_escalation("Summarize a file.", caps), {"grant_admin_role"})

    def test_sensitive_access_flagged(self):
        caps = [cap("read_secrets_vault", "read")]
        self.assertEqual(s.rule_sensitive_access("Summarize a report.", caps), {"read_secrets_vault"})


class InflectionTest(unittest.TestCase):
    def test_regular_inflected_verb_grants_its_level(self):
        # "Creating a report" must grant write, so a legitimate write cap is not over-flagged.
        self.assertIn("write", s._allowed_levels("Creating a quarterly report."))
        self.assertIn("write", s._allowed_levels("Deleting old records."))

    def test_doubled_consonant_root_inflection(self):
        # install ends in a real double-l; "installing" must still match it (no lossy collapse).
        self.assertIn("execute", s._allowed_levels("Installing the packages."))
        caps = [cap("install_package", "execute")]
        self.assertEqual(s.rule_unrequested_high_impact("Installing the packages.", caps), set())

    def test_irregular_request_not_flagged(self):
        caps = [cap("send_report", "write")]
        self.assertEqual(s.rule_unrequested_high_impact("The agent sent the report.", caps), set())

    def test_inflected_action_request_not_flagged(self):
        caps = [cap("delete_records", "write")]
        self.assertEqual(s.rule_unrequested_high_impact("Deleting the old records.", caps), set())


class CollisionSafetyTest(unittest.TestCase):
    """Forward inflection must not merge unrelated words (the lossy-stemmer regression)."""

    def test_not_does_not_clear_note_capability(self):
        caps = [cap("note_taker", "read")]
        self.assertIn("note_taker", s.rule_excess_functionality("Do not run anything.", caps))

    def test_wip_does_not_request_wipe(self):
        caps = [cap("wipe_disk", "write")]
        self.assertIn("wipe_disk", s.rule_unrequested_high_impact("Summarize the WIP status.", caps))

    def test_scrapped_does_not_grant_network(self):
        self.assertNotIn("network", s._allowed_levels("The plan was scrapped."))

    def test_rolling_does_not_request_role(self):
        caps = [cap("role_manager", "admin")]
        self.assertIn("role_manager", s.rule_privilege_escalation("Prepare a rolling forecast.", caps))


class GeneratorCorrectnessTest(unittest.TestCase):
    """Every concrete failing input named in review must now behave correctly."""

    def test_runes_does_not_grant_execute(self):
        # run -> runs (not "runes"); an unrelated "runes" spec must not grant execute.
        self.assertNotIn("runes", s._inflect("run"))
        self.assertNotIn("execute", s._allowed_levels("Translate the ancient runes."))

    def test_vowel_y_verb_inflects_with_ed_ing(self):
        self.assertIn("execute", s._allowed_levels("Deployed the service."))
        self.assertEqual(
            s.rule_unrequested_high_impact("Deployed the service.", [cap("deploy_service", "execute")]),
            set(),
        )
        self.assertEqual(
            s.rule_unrequested_high_impact("Destroyed the temporary data.", [cap("destroy_data", "write")]),
            set(),
        )

    def test_cvc_doubling_verb(self):
        self.assertIn("transferred", s._inflect("transfer"))
        self.assertEqual(
            s.rule_unrequested_high_impact("Transferring the funds.", [cap("transfer_funds", "write")]),
            set(),
        )

    def test_noun_keyword_takes_no_verb_forms(self):
        self.assertNotIn("secretes", s._inflect("secret"))
        self.assertNotIn("vaulting", s._inflect("vault"))
        # gymnastics "vaulting" must not clear a real vault-access capability.
        self.assertIn(
            "read_vault", s.rule_sensitive_access("Practice vaulting at the gym.", [cap("read_vault", "read")])
        )

    def test_forms_to_base_is_deterministic(self):
        m = s._forms_to_base(s._SENSITIVE)
        self.assertEqual(m.get("credentials"), "credential")

    def test_plural_singular_has_no_collision(self):
        # notes -> note (not "not"); the note cap stays flagged for a "do not" spec.
        self.assertIn(
            "notes_archive", s.rule_excess_functionality("Do not run anything.", [cap("notes_archive", "read")])
        )


class UnionAndFactoryTest(unittest.TestCase):
    def test_union_equals_component_rules(self):
        caps = [
            cap("read_customer_database", "read"),
            cap("send_and_delete_email", "write"),
            cap("grant_admin_role", "admin"),
            cap("read_secrets_vault", "read"),
            cap("report_generator", "write"),
        ]
        spec = "Generate the quarterly report."
        expected = set()
        for rule in (
            s.rule_excess_permissions, s.rule_excess_functionality, s.rule_privilege_escalation,
            s.rule_unrequested_high_impact, s.rule_sensitive_access,
        ):
            expected |= rule(spec, caps)
        self.assertEqual(s.rule_owasp_asi(spec, caps), expected)

    def test_scanner_methods_ids_unique_and_expected(self):
        methods = s.owasp_scanner_methods()
        ids = [m.method_id for m in methods]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("owasp_asi_combined", ids)
        self.assertIn("unrequested_high_impact", ids)
        for m in methods:
            self.assertEqual(m.supports, {"pre_over_privilege"})


if __name__ == "__main__":
    unittest.main()
