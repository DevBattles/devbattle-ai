SOLUTION_GENERATOR_PROMPT = """
You are an expert Frontend and Backend AI Engineer.
Given the following coding question:
Title: {title}
Description: {description}
Requirements: {requirements}
Starter Files: {starter_files}
Expected Output: {expected_output}

Generate between 10 and 15 different variations of valid solutions to solve this problem.
Each solution must target a specific implementation paradigm or technology style. Example categories:
- semantic_html_flexbox: Focuses on clean HTML semantics and Flexbox layout
- css_grid: Focuses on CSS Grid design rules
- tailwind_css: Uses Tailwind utility-first styling classes
- React_components: Build with structured functional components and standard hooks
- accessibility_a11y: Emphasizes ARIA roles, tabindex, semantic tags, and screen-readers
- optimized_clean_code: Highly structured, clean variable naming, optimal performance
- custom_paradigm: Any other valid engineering pattern

Also generate a rigorous grading rubric to evaluate student submissions. The rubric must break down criteria by:
1. correctness (0-30 points)
2. responsiveness (0-20 points)
3. accessibility (0-15 points)
4. performance (0-15 points)
5. code_quality (0-20 points)
Total sum is 100 points.

You MUST return a JSON object with this exact schema:
{{
  "solutions": [
    {{
      "type": "paradigm_name (e.g. semantic_html_flexbox)",
      "code": "complete code content resolving the question",
      "explanation": "Brief explanation of this paradigm style"
    }}
  ],
  "rubric": {{
    "correctness": {{
      "max_points": 30,
      "checklist": ["Requirement check 1", "Requirement check 2"]
    }},
    "responsiveness": {{
      "max_points": 20,
      "checklist": ["Requirement check 1"]
    }},
    "accessibility": {{
      "max_points": 15,
      "checklist": ["Requirement check 1"]
    }},
    "performance": {{
      "max_points": 15,
      "checklist": ["Requirement check 1"]
    }},
    "code_quality": {{
      "max_points": 20,
      "checklist": ["Requirement check 1"]
    }}
  }}
}}
Return only valid JSON. Do not include markdown code block syntax.
"""
