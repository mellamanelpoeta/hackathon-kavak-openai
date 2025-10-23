#!/usr/bin/env python3
"""
Validation script to check if the setup is correct.
Run this after installation to verify all components are working.
"""
import sys

def check_imports():
    """Check if all modules can be imported."""
    errors = []

    try:
        from app.models import Customer, Context, Score, Template, InteractionLog
        print("✓ Models imported successfully")
    except Exception as e:
        errors.append(f"✗ Error importing models: {e}")

    try:
        from app.templates import TEMPLATES, get_template_ids
        print("✓ Templates imported successfully")
    except Exception as e:
        errors.append(f"✗ Error importing templates: {e}")

    try:
        from app.scoring import compute_reward
        print("✓ Scoring imported successfully")
    except Exception as e:
        errors.append(f"✗ Error importing scoring: {e}")

    try:
        from app.factories.persona_forge import PersonaForge
        print("✓ PersonaForge imported successfully")
    except Exception as e:
        errors.append(f"✗ Error importing PersonaForge: {e}")

    try:
        from app.factories.state_builder import StateBuilder
        print("✓ StateBuilder imported successfully")
    except Exception as e:
        errors.append(f"✗ Error importing StateBuilder: {e}")

    try:
        from app.factories.template_factory import TemplateFactory
        print("✓ TemplateFactory imported successfully")
    except Exception as e:
        errors.append(f"✗ Error importing TemplateFactory: {e}")

    try:
        from app.factories.policy_learner import ThompsonBandit, EpsilonGreedyBandit
        print("✓ Policy learners imported successfully")
    except Exception as e:
        errors.append(f"✗ Error importing policy learners: {e}")

    try:
        from app.factories.judge import Judge
        print("✓ Judge imported successfully")
    except Exception as e:
        errors.append(f"✗ Error importing Judge: {e}")

    try:
        from app.factories.responders import ResponderAgent, OutreachAgent
        print("✓ Responders imported successfully")
    except Exception as e:
        errors.append(f"✗ Error importing Responders: {e}")

    try:
        from app.factories.prioritizer import Prioritizer
        print("✓ Prioritizer imported successfully")
    except Exception as e:
        errors.append(f"✗ Error importing Prioritizer: {e}")

    try:
        from app.factories.metrics import MetricsAggregator
        print("✓ Metrics imported successfully")
    except Exception as e:
        errors.append(f"✗ Error importing Metrics: {e}")

    try:
        from app.factories.safety import SafetyChecker, ToneValidator
        print("✓ Safety imported successfully")
    except Exception as e:
        errors.append(f"✗ Error importing Safety: {e}")

    return errors

def check_dependencies():
    """Check if all required dependencies are installed."""
    required = [
        'streamlit',
        'pandas',
        'numpy',
        'pydantic',
        'openai',
        'plotly'
    ]

    errors = []
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            errors.append(f"✗ {package} not installed")

    return errors

def check_env():
    """Check environment configuration."""
    import os
    from pathlib import Path

    errors = []

    # Check .env.example exists
    if not Path('.env.example').exists():
        errors.append("✗ .env.example not found")
    else:
        print("✓ .env.example exists")

    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠ OPENAI_API_KEY not set (can be set in UI)")
    else:
        print("✓ OPENAI_API_KEY is set")

    return errors

def main():
    """Run all validation checks."""
    print("=" * 60)
    print("Validating Kavak Demo Setup")
    print("=" * 60)

    print("\n1. Checking dependencies...")
    dep_errors = check_dependencies()

    print("\n2. Checking imports...")
    import_errors = check_imports()

    print("\n3. Checking environment...")
    env_errors = check_env()

    all_errors = dep_errors + import_errors + env_errors

    print("\n" + "=" * 60)
    if all_errors:
        print("❌ Validation failed with errors:")
        for error in all_errors:
            print(f"  {error}")
        print("\nPlease fix the errors and run validation again.")
        sys.exit(1)
    else:
        print("✅ All validation checks passed!")
        print("\nYou can now run the demo:")
        print("  streamlit run app/streamlit_app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
