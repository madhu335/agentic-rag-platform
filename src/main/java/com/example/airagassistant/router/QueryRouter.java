package com.example.airagassistant.router;

import org.springframework.stereotype.Service;

@Service
public class QueryRouter {

    public RouteDecision route(String question) {
        if (question == null || question.isBlank()) {
            return new RouteDecision(Route.RAG, 0.50, "blank_question_default");
        }

        String q = question.toLowerCase().trim();

        if (containsAny(q, "compare", "difference", "vs")) {
            return new RouteDecision(Route.AGENT, 0.90, "comparison_question");
        }

        if (containsAny(q, "why", "how", "explain", "steps")) {
            return new RouteDecision(Route.AGENT, 0.75, "reasoning_or_explanatory_question");
        }

        if (containsAny(q, "what is", "define", "@", "annotation", "bean", "transactional")) {
            return new RouteDecision(Route.RAG, 0.80, "document_friendly_fact_question");
        }

        return new RouteDecision(Route.RAG, 0.60, "default_rag");
    }

    private boolean containsAny(String text, String... terms) {
        for (String term : terms) {
            if (text.contains(term)) {
                return true;
            }
        }
        return false;
    }

    public enum Route {
        RAG,
        AGENT
    }
}