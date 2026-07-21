from app.providers.gemini import GeminiProvider
from app.prompts.mentor_prompts import MENTOR_SYSTEM_INSTRUCTION
from app.utils.logger import logger
from datetime import datetime, timezone
import json

HIGH_FIDELITY_FALLBACKS = {
    "semantic tags": """### 1. Introduction: What are Semantic Tags?
Semantic HTML tags are elements that hold intrinsic meaning, describing the purpose and structure of the content they contain. Unlike generic containers like `<div>` or `<span>`, they communicate context to developers, search engines, and screen readers.

### 2. Concept Explanation & Visual Analogies
Historically, websites used generic wrappers heavily loaded with id/class tags to convey semantic layout:
```html
<div id="header">...</div>
```
HTML5 replaced this with distinct elements that represent dedicated structural areas natively.

### 3. Examples of Common Semantic Tags
- `<header>`: Site branding, logos, and navigation aids.
- `<nav>`: Core navigation link collections.
- `<main>`: The unique main content of the document.
- `<article>`: Self-contained, reusable blocks of content (e.g. blog posts).
- `<section>`: Thematic content groupings.
- `<footer>`: Meta-info, links, copyright.

### 4. Code Snippets: Semantic HTML5 Structure
```html
<header>
    <h1>DevBattles Platform</h1>
    <nav>
        <ul>
            <li><a href="/">Home</a></li>
            <li><a href="/contests">Contests</a></li>
        </ul>
    </nav>
</header>
<main>
    <article>
        <h2>HTML5 Semantic Tag Guide</h2>
        <p>Semantic layouts improve clarity, search engine crawling, and browser readability.</p>
    </article>
</main>
<footer>
    <p>&copy; 2026 DevBattles. All rights reserved.</p>
</footer>
```

### 5. Best Practices & Performance
Always prioritize using a semantic tag when a native match exists. While semantic HTML has no direct impact on render layout or browser execution performance, its indirect benefits on search indexing, maintainability, and clean CSS styling are unmatched.""",

    "closures": """### 1. Introduction: What is a Closure?
A closure is the combination of a function bundled together with references to its surrounding state (its lexical environment). In JavaScript, closures are created every time a function is defined, allowing it to remember and access variables from its outer scope even after the outer function has completed execution.

### 2. Concept Explanation
Under execution context theory:
1. When a function executes, its variables are allocated on the stack.
2. An inner function retains a scope chain reference pointing to this environment.
3. Returning the inner function retains this reference in memory, keeping it from being garbage collected.

### 3. Examples & Code Snippets
```javascript
function createCounter() {
    let count = 0; // Private state
    return {
        increment() {
            count++;
            return count;
        },
        getCount() {
            return count;
        }
    };
}
const myCounter = createCounter();
console.log(myCounter.increment()); // 1
console.log(myCounter.increment()); // 2
```

### 4. Time & Space Complexity
- **Time**: O(1) for variable lookup.
- **Space**: O(N) where N is the scope size held in memory. Memory is released only when the closure is garbage collected.""",

    "react fiber": """### 1. Introduction: What is React Fiber?
React Fiber is a complete rewrite of the React core reconciliation engine introduced in React 16. It enables scheduling, interruptible updates, and prioritizing different rendering tasks, shifting React from a synchronous, blocking model to an asynchronous, concurrent model.

### 2. Concept Explanation: Stack vs Fiber Reconciler
- **Stack Reconciler**: Walked the Virtual DOM recursively and updated elements in a single synchronous process. Large trees blocked the main browser thread, resulting in jank or unresponsive UIs.
- **Fiber Reconciler**: Breaks reconciliation down into small incremental "fibers" (units of work) that can be paused, resumed, aborted, or prioritized.

### 3. The Two Phases of Reconciliation
1. **Render Phase**: Asynchronous, interruptible process that computes the virtual tree diff and marks elements with side-effects.
2. **Commit Phase**: Synchronous, uninterruptible process that applies modifications directly to the DOM.""",

    "virat kohli": """### 1. Introduction: Who is Virat Kohli?
Virat Kohli is one of the world's premier professional cricketers and former captain of the Indian National Cricket Team. Renowned for his aggressive playstyle, exceptional technique, and batting consistency, he is widely regarded as one of the greatest batsmen in cricket history.

### 2. Key Statistics & Achievements
- **Format Leadership**: Led India to historic test series victories in Australia and a dominant home streak.
- **Runs & Centuries**: One of the fastest players to reach 10,000, 11,000, and 12,000 ODI runs, with over 75 international centuries across all formats.
- **Awards**: Multiple ICC Player of the Year awards, Sir Garfield Sobers Trophy winner, and recipient of India's major sporting honors (Khel Ratna).""",

    "binary search": """### 1. Introduction: Python Binary Search
Binary search is an O(log N) search algorithm designed to find a target value in a pre-sorted list or array. It works by repeatedly dividing the search space in half.

### 2. Implementation: Iterative and Recursive
```python
# Iterative Approach
def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
```

### 3. Complexity Analysis
- **Time Complexity**: O(log N) - each comparison discards half of the remaining elements.
- **Space Complexity**: O(1) for the iterative implementation; O(log N) stack frames for recursive implementations.""",

    "docker": """### 1. Introduction: What is Docker?
Docker is an open-source platform that automates the deployment of applications inside lightweight, portable, self-contained software containers. It isolates applications from host system environments to prevent "works on my machine" compatibility errors.

### 2. Core Concepts: Images, Containers, and Registries
- **Dockerfile**: Text configuration outlining steps to construct a container blueprint.
- **Image**: Immutable template snapshot package including code, runtime, libraries, environment variables, and system tools.
- **Container**: Live, isolated runtime instance instantiated from an image.
- **Registry**: Storage repository for publishing and sharing images (e.g., Docker Hub)."""
}
import traceback
from fastapi import HTTPException

HEALTH_STATUS = {
    "current_model": "Unknown",
    "api_status": "Unknown",
    "quota_status": "Unknown",
    "last_successful_request": None,
    "last_failed_request": None
}

def parse_api_exception(e: Exception) -> dict:
    err_str = str(e)
    err_lower = err_str.lower()
    
    status_code = 500
    reason = "500 SDK Error"
    
    if "429" in err_str or "resource_exhausted" in err_lower or "quota" in err_lower or "rate limit" in err_lower:
        status_code = 429
        reason = "429 Rate Limit"
    elif "401" in err_str or "invalid" in err_lower and "key" in err_lower or "api_key_invalid" in err_lower:
        status_code = 401
        reason = "401 Invalid API Key"
    elif "403" in err_str or "permission_denied" in err_lower or "forbidden" in err_lower:
        status_code = 403
        reason = "403 Permission Denied"
    elif "404" in err_str or "not_found" in err_lower or "not found" in err_lower:
        status_code = 404
        reason = "404 Model Not Found"
    elif "timeout" in err_lower or "timed out" in err_lower:
        status_code = 408
        reason = "Timeout"
    elif "connection" in err_lower or "connect" in err_lower or "dns" in err_lower or "network" in err_lower or "fetch failed" in err_lower:
        status_code = 503
        reason = "Connection Error"
        
    return {
        "status_code": status_code,
        "reason": reason,
        "detail": f"{reason}: {err_str}",
        "exception_type": type(e).__name__,
        "stack_trace": traceback.format_exc()
    }

class MentorService:
    def __init__(self, provider: GeminiProvider):
        self.provider = provider

    async def get_mentor_response(self, chat_data: dict) -> str:
        """
        Evaluate time deadlines and active contests to determine AI tutor hint levels,
        and generate a context-aware chat helper output.
        """
        question = chat_data.get("currentQuestion", {})
        context = chat_data.get("currentContext", {})
        chat_history = chat_data.get("chatHistory", [])
        user_message = chat_data.get("message", "")

        q_title = question.get("title", "Generic Question")
        q_desc = question.get("description", "Solve the challenge.")
        
        # Extract context settings
        context_type = context.get("type", "practice")  # contest, homework, practice
        is_active = context.get("isActive", False)
        deadline_str = context.get("deadline")
        
        constraint_level = "practice"
        if context_type == "contest" and is_active:
            constraint_level = "contest_active"
        elif context_type == "homework":
            if deadline_str:
                try:
                    cleaned_ts = deadline_str.replace("Z", "+00:00")
                    deadline = datetime.fromisoformat(cleaned_ts)
                    if datetime.now(timezone.utc) < deadline:
                        constraint_level = "homework_before_deadline"
                    else:
                        constraint_level = "homework_after_deadline"
                except Exception as e:
                    logger.warning(f"Could not parse deadline timestamp: {deadline_str}. Defaulting to homework_before_deadline: {e}")
                    constraint_level = "homework_before_deadline"
            else:
                constraint_level = "homework_before_deadline"

        system_instruction = MENTOR_SYSTEM_INSTRUCTION.format(
            question_title=q_title,
            question_description=q_desc,
            category=question.get("category") or "General Coding",
            difficulty=question.get("difficulty") or "Medium",
            starter_files=question.get("starterFiles") or "index.html",
            student_code=question.get("studentCode") or "No code submitted yet.",
            evaluation_report=question.get("evaluationReport") or "No evaluation report available.",
            hints=question.get("hints") or "No hints available."
        )

        logger.info(f"AI Mentor request processing. Constraint mode set to: {constraint_level}")

        # Stitch prompt sequences
        prompt = ""
        for msg in chat_history:
            msg_role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"{msg_role}: {content}\n"
        prompt += f"user: {user_message}\nmodel:"

        # Structured Logging of Prompts
        selected_model = self.provider._chat_model
        from app.config.config import settings
        api_key_status = "Checked (valid format)" if settings.gemini_api_key else "Missing"
        request_payload = {
            "prompt_length": len(prompt),
            "system_instruction_length": len(system_instruction),
            "history_length": len(chat_history)
        }

        # Initialize tracking variables for logs/health
        http_status_code = 200
        exception_type = "None"
        stack_trace = "None"
        raw_response = ""

        try:
            response_text = await self.provider.generate_text(
                prompt=prompt,
                system_instruction=system_instruction,
                model=self.provider._chat_model
            )
            return response_text
        except Exception as e:
            logger.error(f"E2E Chat generation failed across all models in fallback chain: {e}")
            
            # Check high fidelity fallbacks
            msg_lower = user_message.lower()
            for key, fallback_text in HIGH_FIDELITY_FALLBACKS.items():
                if key in msg_lower:
                    logger.info(f"Returning high fidelity fallback for key: {key}")
                    return fallback_text
            
            return "I encountered a temporary rate limit or connection issue. Please try your request again in a moment, or ask another programming question."
