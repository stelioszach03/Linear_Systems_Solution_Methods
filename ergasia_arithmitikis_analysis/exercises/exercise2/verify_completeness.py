"""
Verification script to ensure all required components of Exercise 2 are complete
and properly documented.
"""
import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    exists = os.path.exists(filepath)
    print(f"[{'✓' if exists else '✗'}] {description}: {filepath}")
    return exists

def check_file_content(filepath, required_elements):
    if not os.path.exists(filepath):
        return False

    with open(filepath, 'r') as f:
        content = f.read().lower()

    all_found = True
    for element, description in required_elements:
        found = element.lower() in content
        print(f"  [{'✓' if found else '✗'}] {description}")
        all_found = all_found and found

    return all_found

def main():
    exercise_dir = Path(__file__).parent

    # Required files checklist
    required_files = [
        ('run_timing_comparison.py', 'Main timing comparison implementation'),
        ('timing_analysis_results.txt', 'Detailed timing results'),
        ('theoretical_analysis.txt', 'Theoretical justification'),
        ('detailed_solution_steps.txt', 'Step-by-step solution process'),
        ('plot_timing_results.py', 'Performance visualization script'),
        ('timing_comparison_plots.png', 'Performance comparison plots'),
        ('plot_description.txt', 'Plot documentation'),
        ('README.md', 'Project documentation')
    ]

    # Content requirements for key files
    theoretical_requirements = [
        ('O(n)', 'Complexity analysis for Thomas method'),
        ('O(n³)', 'Complexity analysis for Gaussian elimination'),
        ('memory', 'Memory usage analysis'),
        ('cache', 'Cache efficiency discussion')
    ]

    detailed_steps_requirements = [
        ('forward elimination', 'Forward elimination process'),
        ('back substitution', 'Back substitution process'),
        ('multiplier', 'Multiplier calculations'),
        ('intermediate', 'Intermediate steps')
    ]

    readme_requirements = [
        ('usage', 'Usage instructions'),
        ('implementation', 'Implementation details'),
        ('verification', 'Verification process'),
        ('results', 'Results summary')
    ]

    print("\nExercise 2 Completeness Verification")
    print("===================================")

    # Check all required files
    print("\n1. Required Files Check:")
    all_files_exist = True
    for filename, description in required_files:
        file_exists = check_file_exists(exercise_dir / filename, description)
        all_files_exist = all_files_exist and file_exists

    # Check content requirements
    print("\n2. Content Requirements Check:")

    print("\n2.1 Theoretical Analysis:")
    theoretical_complete = check_file_content(
        exercise_dir / 'theoretical_analysis.txt',
        theoretical_requirements
    )

    print("\n2.2 Detailed Steps:")
    steps_complete = check_file_content(
        exercise_dir / 'detailed_solution_steps.txt',
        detailed_steps_requirements
    )

    print("\n2.3 Documentation:")
    docs_complete = check_file_content(
        exercise_dir / 'README.md',
        readme_requirements
    )

    # Final verification
    all_complete = all_files_exist and theoretical_complete and steps_complete and docs_complete

    print("\nVerification Summary")
    print("===================")
    print(f"Required files present: {'✓' if all_files_exist else '✗'}")
    print(f"Theoretical analysis complete: {'✓' if theoretical_complete else '✗'}")
    print(f"Detailed steps documented: {'✓' if steps_complete else '✗'}")
    print(f"Documentation requirements met: {'✓' if docs_complete else '✗'}")
    print(f"\nOverall status: {'✓ COMPLETE' if all_complete else '✗ INCOMPLETE'}")

    if not all_complete:
        print("\nMissing requirements:")
        if not all_files_exist:
            print("- Some required files are missing")
        if not theoretical_complete:
            print("- Theoretical analysis needs additional content")
        if not steps_complete:
            print("- Detailed steps documentation incomplete")
        if not docs_complete:
            print("- README.md missing required sections")

if __name__ == "__main__":
    main()
