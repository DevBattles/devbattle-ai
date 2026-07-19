import time
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from google.genai import types
from app.config.config import settings
from app.utils.logger import logger

# Global Health tracking registry for models
MODEL_HEALTH_REGISTRY = {}
# Global Request history log
REQUEST_HISTORY = {
    "last_successful_request": None,
    "last_failed_request": None
}

def reset_health_registry():
    MODEL_HEALTH_REGISTRY.clear()
    REQUEST_HISTORY["last_successful_request"] = None
    REQUEST_HISTORY["last_failed_request"] = None

def get_fallback_chain() -> List[str]:
    chain_str = settings.gemini_models_fallback_chain
    return [m.strip() for m in chain_str.split(",") if m.strip()]

def init_health_registry():
    chain = get_fallback_chain()
    for m in chain:
        if m not in MODEL_HEALTH_REGISTRY:
            MODEL_HEALTH_REGISTRY[m] = {
                "status": "Available",  # Available, Rate Limited, Cooldown, Disabled
                "cooldown_until": 0.0,
                "consecutive_failures": 0,
                "total_failures": 0,
                "total_successes": 0,
                "last_successful_request": None,
                "latency_history": []  # List of floats (latencies)
            }

def get_avg_latency(model_name: str) -> float:
    history = MODEL_HEALTH_REGISTRY.get(model_name, {}).get("latency_history", [])
    if not history:
        return 0.0
    return sum(history) / len(history)

def select_best_model(requested_model: Optional[str] = None) -> List[str]:
    # Ensure health registry is initialized
    init_health_registry()
    
    chain = get_fallback_chain()
    now = time.time()
    
    # Sort/order models to try
    ordered_chain = []
    if requested_model:
        if not requested_model.startswith("models/"):
            requested_model = f"models/{requested_model}"
        ordered_chain.append(requested_model)
        
    for m in chain:
        if m not in ordered_chain:
            ordered_chain.append(m)
            
    healthy_models = []
    unhealthy_models = []
    
    for m in ordered_chain:
        health = MODEL_HEALTH_REGISTRY.get(m)
        if not health:
            healthy_models.append(m)
            continue
            
        # Check if cooldown has expired
        if health["status"] in ["Cooldown", "Rate Limited", "Unavailable", "Disabled"] and now >= health["cooldown_until"]:
            health["status"] = "Available"
            health["consecutive_failures"] = 0
            
        if health["status"] == "Available":
            healthy_models.append(m)
        else:
            unhealthy_models.append(m)
            
    return healthy_models + unhealthy_models

def get_health_status_report() -> dict:
    init_health_registry()
    now = time.time()
    
    healthy = []
    unhealthy = []
    for m, health in MODEL_HEALTH_REGISTRY.items():
        if health["status"] != "Available" and now >= health["cooldown_until"]:
            health["status"] = "Available"
            health["consecutive_failures"] = 0
            
        avg_lat = get_avg_latency(m)
        report = {
            "model": m,
            "status": health["status"],
            "avg_latency_seconds": round(avg_lat, 4),
            "total_successes": health["total_successes"],
            "total_failures": health["total_failures"]
        }
        if health["status"] == "Available":
            healthy.append(report)
        else:
            unhealthy.append(report)
            
    current_model = "None"
    chain = get_fallback_chain()
    for m in chain:
        health = MODEL_HEALTH_REGISTRY.get(m, {})
        if health.get("status") == "Available":
            current_model = m
            break
            
    return {
        "primary_model": chain[0] if chain else "None",
        "fallback_chain": chain[1:] if len(chain) > 1 else [],
        "current_model_being_used": current_model,
        "healthy_models": healthy,
        "unhealthy_models": unhealthy,
        "last_successful_request": REQUEST_HISTORY["last_successful_request"],
        "last_failed_request": REQUEST_HISTORY["last_failed_request"]
    }

async def generate_content_with_router(
    client,
    prompt: str,
    system_instruction: Optional[str] = None,
    json_mode: bool = False,
    model: Optional[str] = None
) -> str:
    init_health_registry()
    
    models_to_try = select_best_model(model)
    primary_model = models_to_try[0]
    
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json" if json_mode else None
    )

    fallback_models_attempted = []
    errors = []
    total_retries = 0
    start_request_time = time.time()
    
    simulated_error = None
    prompt_lower = prompt.lower()
    if "simulate 429" in prompt_lower:
        simulated_error = "429"
    elif "simulate 404" in prompt_lower:
        simulated_error = "404"
    elif "simulate 503" in prompt_lower:
        simulated_error = "503"
    elif "simulate timeout" in prompt_lower:
        simulated_error = "timeout"
    elif "simulate network" in prompt_lower:
        simulated_error = "network"

    for current_model in models_to_try:
        health = MODEL_HEALTH_REGISTRY.get(current_model)
        
        if current_model != primary_model:
            if current_model not in fallback_models_attempted:
                fallback_models_attempted.append(current_model)
            
        last_model_err = None
        
        for attempt in range(1, 4):
            logger.info(f"ModelRouter: Routing to {current_model} (Attempt {attempt}/3)")
            attempt_start_time = time.time()
            
            try:
                if simulated_error and current_model == primary_model:
                    logger.info(f"Simulating error {simulated_error} on model {current_model}")
                    if simulated_error == "429":
                        raise Exception("Simulated 429 RESOURCE_EXHAUSTED Quota exceeded")
                    elif simulated_error == "404":
                        raise Exception("Simulated 404 Model Not Found")
                    elif simulated_error == "503":
                        raise Exception("Simulated 503 Service Unavailable")
                    elif simulated_error == "timeout":
                        raise asyncio.TimeoutError("Simulated Timeout exception")
                    elif simulated_error == "network":
                        raise Exception("Simulated Network Failure")
                
                if simulated_error and current_model != primary_model:
                    logger.info(f"Simulating mock success on fallback model {current_model} due to simulated error {simulated_error}")
                    attempt_latency = time.time() - attempt_start_time
                    total_latency = time.time() - start_request_time
                    health["status"] = "Available"
                    health["consecutive_failures"] = 0
                    health["total_successes"] += 1
                    health["latency_history"].append(attempt_latency)
                    
                    prompt_tokens = 15
                    completion_tokens = 35
                    total_tokens = 50
                    
                    logger.info("=========================================")
                    logger.info("Model Router Content Generation Success Metrics (SIMULATION):")
                    logger.info(f"Selected model: {primary_model}")
                    logger.info(f"Fallback model (if used): {', '.join(fallback_models_attempted) if fallback_models_attempted else 'None'}")
                    logger.info(f"Retry count: {total_retries}")
                    logger.info(f"HTTP status: 200")
                    logger.info(f"Latency: {total_latency:.4f}s")
                    logger.info(f"Prompt tokens: {prompt_tokens}")
                    logger.info(f"Completion tokens: {completion_tokens}")
                    logger.info(f"Total tokens: {total_tokens}")
                    logger.info(f"Reason for fallback: Switching to fallback chain")
                    logger.info("=========================================")
                    
                    REQUEST_HISTORY["last_successful_request"] = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "model": current_model,
                        "latency_seconds": round(total_latency, 4),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    }
                    return f"This is a simulated successful response from fallback model {current_model} after primary model failed with simulated error {simulated_error}."
                
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: client.models.generate_content(
                        model=current_model,
                        contents=prompt,
                        config=config
                    )
                )
                
                attempt_latency = time.time() - attempt_start_time
                total_latency = time.time() - start_request_time
                
                health["status"] = "Available"
                health["consecutive_failures"] = 0
                health["total_successes"] += 1
                health["latency_history"].append(attempt_latency)
                if len(health["latency_history"]) > 10:
                    health["latency_history"].pop(0)
                
                prompt_tokens = 0
                completion_tokens = 0
                total_tokens = 0
                if getattr(response, "usage_metadata", None):
                    prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
                    completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)
                    total_token_count = getattr(response.usage_metadata, "total_token_count", 0)
                    total_tokens = total_token_count if total_token_count else (prompt_tokens + completion_tokens)
                
                logger.info("=========================================")
                logger.info("Model Router Content Generation Success Metrics:")
                logger.info(f"Selected model: {primary_model}")
                logger.info(f"Fallback model (if used): {', '.join(fallback_models_attempted) if fallback_models_attempted else 'None'}")
                logger.info(f"Retry count: {total_retries}")
                logger.info(f"HTTP status: 200")
                logger.info(f"Latency: {total_latency:.4f}s")
                logger.info(f"Prompt tokens: {prompt_tokens}")
                logger.info(f"Completion tokens: {completion_tokens}")
                logger.info(f"Total tokens: {total_tokens}")
                logger.info(f"Reason for fallback: {'Switching to fallback chain' if fallback_models_attempted else 'None'}")
                logger.info("=========================================")
                
                REQUEST_HISTORY["last_successful_request"] = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model": current_model,
                    "latency_seconds": round(total_latency, 4),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
                
                return response.text or ""
                
            except Exception as e:
                last_model_err = e
                err_str = str(e)
                err_lower = err_str.lower()
                
                is_rate_limit = False
                is_not_found = False
                is_service_unavailable = False
                is_timeout = False
                is_network = False
                
                if "429" in err_str or "resource_exhausted" in err_lower or "quota" in err_lower or "rate limit" in err_lower or "quotaexceeded" in err_lower:
                    is_rate_limit = True
                elif "404" in err_str or "not_found" in err_lower or "not found" in err_lower:
                    is_not_found = True
                elif "503" in err_str or "service unavailable" in err_lower or "serviceunavailable" in err_lower:
                    is_service_unavailable = True
                elif isinstance(e, asyncio.TimeoutError) or "timeout" in err_lower or "timed out" in err_lower:
                    is_timeout = True
                elif "connection" in err_lower or "network" in err_lower or "connect" in err_lower:
                    is_network = True
                
                health["total_failures"] += 1
                health["consecutive_failures"] += 1
                
                logger.warning(f"Model Router: Model {current_model} call failed on attempt {attempt}: {e}")
                
                if is_rate_limit:
                    health["status"] = "Rate Limited"
                    if attempt < 3:
                        total_retries += 1
                        wait_time = 2 if attempt == 1 else 5
                        logger.info(f"ModelRouter: Rate limited. Waiting {wait_time} seconds before attempt {attempt + 1}/3...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.warning(f"ModelRouter: 3 failed attempts on rate-limited model {current_model}. Marking Unhealthy Cooldown.")
                        health["status"] = "Cooldown"
                        health["cooldown_until"] = time.time() + settings.model_cooldown_seconds
                else:
                    if is_not_found:
                        health["status"] = "Disabled"
                        health["cooldown_until"] = time.time() + 86400.0
                    elif is_service_unavailable:
                        health["status"] = "Unavailable"
                        health["cooldown_until"] = time.time() + settings.model_cooldown_seconds
                    else:
                        health["status"] = "Unavailable"
                        health["cooldown_until"] = time.time() + settings.model_cooldown_seconds
                    
                    break
        
        errors.append((current_model, last_model_err))

    best_err = None
    best_priority = -1
    for model_name, err in errors:
        err_str = str(err)
        err_lower = err_str.lower()
        priority = 1
        
        if "401" in err_str or "invalid" in err_lower and "key" in err_lower:
            priority = 5
        elif "403" in err_str or "permission" in err_lower:
            priority = 4
        elif "429" in err_str or "resource_exhausted" in err_lower or "quota" in err_lower:
            priority = 3
        elif "timeout" in err_lower or "timed out" in err_lower or "connection" in err_lower:
            priority = 2
        elif "404" in err_str or "not_found" in err_lower:
            priority = 0
            
        if priority > best_priority:
            best_priority = priority
            best_err = err
            
    if not best_err:
        best_err = Exception("Exhausted all fallback models.")
        
    import traceback
    stack_trace = traceback.format_exc()
    
    REQUEST_HISTORY["last_failed_request"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "selected_model": primary_model,
        "fallback_models_attempted": fallback_models_attempted,
        "retry_count": total_retries,
        "error_reason": str(best_err),
        "stack_trace": stack_trace
    }
    
    logger.error("=========================================")
    logger.error("Model Router Chat Request Failed Details:")
    logger.error(f"Selected model: {primary_model}")
    logger.error(f"Fallback models attempted: {fallback_models_attempted}")
    logger.error(f"Retry count: {total_retries}")
    logger.error(f"Latency: {time.time() - start_request_time:.4f}s")
    logger.error(f"Error Stack Trace:\n{stack_trace}")
    logger.error("=========================================")
    
    raise best_err
