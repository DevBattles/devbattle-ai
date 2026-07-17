SOLUTION_GENERATOR_PROMPT = """
You are an expert Frontend and Backend AI Engineer.
Given the following coding question:
Title: {title}
Description: {description}
Requirements: {requirements}
Starter Files: {starter_files}
Expected Output: {expected_output}

You MUST perform Question Analysis and Classification:
1. Classify the question into one of the following exact categories:
   HTML, CSS, JavaScript, TypeScript, React, Next.js, Node.js, Express, MongoDB, SQL, Python, Java, C++, C, Data Structures, Algorithms, Debugging, Output Prediction, MCQ, Theory, System Design, API Design, JSON, Git, Linux, Docker, DevOps, Mixed Project, Frontend Project, Backend Project, Full Stack Project.
2. Select the appropriate Workspace Configuration:
   - "mcq": No Editor, Options select UI. (Used for Multiple Choice Questions)
   - "theory": Rich Text/Markdown Editor, No visual preview.
   - "javascript_logic": main.js, Console Output, No visual preview. (Used for JS logic like reversing strings, calculating factorials, data structures)
   - "python": main.py, Console Output, No visual preview.
   - "cpp": main.cpp, Console Output, No visual preview.
   - "sql": query.sql, Console Output, No visual preview, Execute SQL engine.
   - "react": App.jsx, main.jsx, package.json, Visual browser preview.
   - "html": index.html, style.css, script.js, Visual browser preview.
   - "css": index.html, style.css, Visual browser preview.
   - "node": server.js, package.json, Console Output, No visual preview.
   - "express": server.js, routes, controllers, Console Output, No visual preview.
3. Select the appropriate parameters:
   - evaluationStrategy: "compare_answers" (for MCQ), "semantic_llm" (for Theory), "run_tests" (for JS/Python/Node/Express), "compile_and_run" (for C++), "execute_sql" (for SQL), "ui_playwright" (for React/HTML/CSS).
   - supportedLanguage: e.g., "javascript", "python", "cpp", "sql", "html", "css", "markdown", "text".
   - previewRequired: true or false. (Set to true only for React, HTML, CSS visual UI projects; false for SQL, JS, Python, C++, Node, MCQ, Theory).
   - executionMode: "console", "browser", or "none".
4. Populate "options":
   - For MCQ: extract or generate 4 choices (e.g. `["Option A text", "Option B text", "Option C text", "Option D text"]`) from the question description/requirements.
   - For others: return null or an empty list `[]`.
5. Generate starterFiles:
   - If the teacher did not supply specific files or only supplied an empty index.html for a non-HTML category, generate clean starter code files matching the workspace configuration (e.g., a starter `main.js` with functional headers for JavaScript, `query.sql` for SQL, `main.py` for Python, `main.cpp` for C++).
6. Generate 10 to 15 different variations of valid solutions to solve this problem matching the starter files structure.
7. Generate a rigorous grading rubric summing up to 100 points matching these weights:
   - correctness (0-40 points)
   - edge_cases (0-20 points)
   - requirements (0-20 points)
   - code_quality (0-10 points)
   - performance (0-10 points)

You MUST return a JSON object with this exact schema:
{{
  "category": "category name",
  "workspaceType": "workspace_type_name",
  "evaluationStrategy": "evaluation_strategy_name",
  "supportedLanguage": "supported_language_name",
  "previewRequired": true_or_false,
  "executionMode": "execution_mode_name",
  "options": ["A", "B", "C", "D"]_or_null,
  "starterFiles": {{
     "filename.ext": {{ "content": "starter code content" }}
  }},
  "solutions": [
    {{
      "type": "paradigm_name",
      "code": "complete code content resolving the question",
      "explanation": "brief explanation"
    }}
  ],
  "rubric": {{
    "correctness": {{ "max_points": 40, "checklist": [...] }},
    "edge_cases": {{ "max_points": 20, "checklist": [...] }},
    "requirements": {{ "max_points": 20, "checklist": [...] }},
    "code_quality": {{ "max_points": 10, "checklist": [...] }},
    "performance": {{ "max_points": 10, "checklist": [...] }}
  }}
}}
Return only valid JSON. Do not include markdown code block syntax.
"""
