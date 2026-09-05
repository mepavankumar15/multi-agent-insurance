"""Test script for all 3 claim scenarios."""
from claims_agents import process_claim, SAMPLE_CLAIMS

for scenario, raw in SAMPLE_CLAIMS.items():
    print("=" * 60)
    print(f"  SCENARIO: {scenario.upper()}")
    print("=" * 60)
    
    result = process_claim(raw)
    
    print(f"Claimant:       {result.get('claimant_name')}")
    print(f"Policy:         {result.get('policy_number')}")
    print(f"Claim Type:     {result.get('claim_type')}")
    print(f"Amount:         ${result.get('claim_amount', 0):,.2f}")
    print(f"Coverage Limit: ${result.get('coverage_limit', 0):,.2f}")
    print(f"Policy Valid:   {result.get('coverage_valid')}")
    print(f"Fraud Score:    {result.get('fraud_score')}/100")
    print(f"Red Flags:      {result.get('red_flags')}")
    print(f"Decision:       {result.get('decision_status')}")
    print(f"Reason:         {result.get('decision_reason')}")
    print()
    print("Trace:")
    for t in result.get("trace", []):
        print(f"  {t}")
    print()
