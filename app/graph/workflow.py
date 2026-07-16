from langgraph.graph import StateGraph, START, END
from app.graph.state import SubmissionState
from app.graph.nodes import (
    validate_input_node,
    retrieve_question_node,
    retrieve_rubric_node,
    retrieve_similar_solutions_node,
    vision_check_node,
    gemini_evaluate_node,
    aggregate_scores_node,
    generate_report_node
)

def build_workflow():
    workflow = StateGraph(SubmissionState)

    # Register nodes
    workflow.add_node("validate_input", validate_input_node)
    workflow.add_node("retrieve_question", retrieve_question_node)
    workflow.add_node("retrieve_rubric", retrieve_rubric_node)
    workflow.add_node("retrieve_similar_solutions", retrieve_similar_solutions_node)
    workflow.add_node("vision_check", vision_check_node)
    workflow.add_node("gemini_evaluate", gemini_evaluate_node)
    workflow.add_node("aggregate_scores", aggregate_scores_node)
    workflow.add_node("generate_report", generate_report_node)

    # Build routing edges
    workflow.add_edge(START, "validate_input")
    
    # Check for early error exits
    def route_after_validation(state):
        if state.get("error"):
            return "generate_report"
        return "retrieve_question"

    workflow.add_conditional_edges(
        "validate_input",
        route_after_validation,
        {
            "generate_report": "generate_report",
            "retrieve_question": "retrieve_question"
        }
    )

    workflow.add_edge("retrieve_question", "retrieve_rubric")
    workflow.add_edge("retrieve_rubric", "retrieve_similar_solutions")
    workflow.add_edge("retrieve_similar_solutions", "vision_check")
    workflow.add_edge("vision_check", "gemini_evaluate")
    workflow.add_edge("gemini_evaluate", "aggregate_scores")
    workflow.add_edge("aggregate_scores", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile()

app_workflow = build_workflow()
