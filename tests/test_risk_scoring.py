"""Tests for the rule-based risk scoring logic."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import risk_scoring


def test_auto_renewal_without_notice_is_high_risk():
    text = "This Agreement shall automatically renew for successive one-year terms unless the Company provides written notice."
    result = risk_scoring.assess(text, ["Renewal Term"])
    assert result[0]["level"] == "High"


def test_auto_renewal_with_clear_notice_is_low_risk():
    text = (
        "This Agreement shall automatically renew for successive one-year terms "
        "unless either party provides written notice of termination at least 90 "
        "days prior to the end of the then-current term."
    )
    result = risk_scoring.assess(text, ["Renewal Term"])
    assert result[0]["level"] == "Low"


def test_notice_period_stated_in_months_is_recognized():
    """Regression test: a real external contract stated notice in months and
    was previously misflagged High because the pattern only recognized days."""
    text = (
        "This Agreement will automatically renew for additional one year terms "
        "unless either Party elects not to renew this Agreement by written "
        "notice to the other Party, which notice must be provided at least six "
        "(6) months prior to the expiration of the Initial Term."
    )
    result = risk_scoring.assess(text, ["Renewal Term"])
    assert result[0]["level"] == "Low"


def test_explicit_uncapped_liability_is_high_risk():
    text = "There shall be no limit on either party's liability for breach of confidentiality obligations under this Agreement."
    result = risk_scoring.assess(text, ["Uncapped Liability"])
    assert result[0]["level"] == "High"


def test_damages_exclusion_clause_is_not_flagged_high():
    """Regression test: a standard damages-exclusion clause containing
    'without limitation' was previously misflagged as uncapped liability."""
    text = (
        "In no event shall either Party be liable to the other Party for any "
        "special, indirect, punitive, or consequential damages of any kind "
        "(including, without limitation, lost profits, business, or goodwill), "
        "regardless of whether such liability is based on breach of contract."
    )
    result = risk_scoring.assess(text, ["Uncapped Liability"])
    assert result[0]["level"] == "Low"


def test_liability_capped_to_short_window_is_medium_risk():
    text = "In no event shall either party's liability exceed the total fees paid in the preceding 12 months."
    result = risk_scoring.assess(text, ["Cap On Liability"])
    assert result[0]["level"] == "Medium"


def test_broad_and_long_noncompete_is_high_risk():
    text = "Employee agrees not to engage in any business that competes with the Company worldwide for a period of 5 years following termination."
    result = risk_scoring.assess(text, ["Non-Compete"])
    assert result[0]["level"] == "High"


def test_narrow_noncompete_is_low_risk():
    text = "For a period of six months following termination, Consultant agrees not to provide similar services to Client's direct competitors within Client's metropolitan market."
    result = risk_scoring.assess(text, ["Non-Compete"])
    assert result[0]["level"] == "Low"


def test_mutual_termination_is_low_risk():
    text = "Either party may terminate this Agreement upon 60 days written notice to the other party."
    result = risk_scoring.assess(text, ["Termination For Convenience"])
    assert result[0]["level"] == "Low"


def test_one_sided_termination_is_medium_risk():
    text = "The Company may terminate this Agreement at any time, with or without cause, upon fourteen days written notice to Employee."
    result = risk_scoring.assess(text, ["Termination For Convenience"])
    assert result[0]["level"] == "Medium"


def test_one_sided_termination_recognizes_other_party_roles():
    """Regression test: a real external contract used 'Manufacturer may
    terminate' rather than 'Company may terminate', which the original
    role-noun list did not recognize."""
    text = "Manufacturer may terminate this Agreement at any time following the First Commercial Sale upon nine months written notice to Distributor."
    result = risk_scoring.assess(text, ["Termination For Convenience"])
    assert result[0]["level"] == "Medium"


def test_unrestricted_audit_is_medium_risk():
    text = "Company may audit Distributor's records at any time without prior notice."
    result = risk_scoring.assess(text, ["Audit Rights"])
    assert result[0]["level"] == "Medium"


def test_reasonable_audit_is_low_risk():
    text = "Company may audit Distributor's records once annually upon reasonable notice during normal business hours."
    result = risk_scoring.assess(text, ["Audit Rights"])
    assert result[0]["level"] == "Low"


def test_insurance_without_amount_is_medium_risk():
    text = "Licensee shall obtain and maintain comprehensive general liability insurance."
    result = risk_scoring.assess(text, ["Insurance"])
    assert result[0]["level"] == "Medium"


def test_insurance_with_specific_amount_is_low_risk():
    text = "Licensee shall maintain liability insurance with coverage of at least $2 million per occurrence."
    result = risk_scoring.assess(text, ["Insurance"])
    assert result[0]["level"] == "Low"


def test_broad_indefinite_exclusivity_is_high_risk():
    text = "Distributor shall have the exclusive worldwide right to distribute the Products in perpetuity."
    result = risk_scoring.assess(text, ["Exclusivity"])
    assert result[0]["level"] == "High"


def test_disclaimed_warranty_is_high_risk():
    text = "The Products are provided AS-IS and Manufacturer disclaims all warranties."
    result = risk_scoring.assess(text, ["Warranty Duration"])
    assert result[0]["level"] == "High"


def test_volume_restriction_with_penalty_is_medium_risk():
    text = "Distributor shall purchase a minimum of 10,000 units annually; failure to meet this minimum shall result in a shortfall penalty."
    result = risk_scoring.assess(text, ["Volume Restriction"])
    assert result[0]["level"] == "Medium"


def test_unknown_category_returns_no_result():
    result = risk_scoring.assess("Some clause text.", ["Not A Real Category"])
    assert result == []


def test_empty_category_list_returns_no_result():
    result = risk_scoring.assess("Some clause text.", [])
    assert result == []
