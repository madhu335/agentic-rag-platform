package com.example.airagassistant.agentic.multi;

import com.example.airagassistant.agentic.dto.AgentRequest;
import com.example.airagassistant.agentic.dto.AgentResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * HTTP entry point for the multi-agent supervisor flow.
 *
 * Endpoint: POST /agent/multi
 * Request body: same AgentRequest as POST /agent
 * Response body: same AgentResponse as POST /agent
 *
 * This coexists with the existing POST /agent (single-agent) endpoint.
 * Both use the same request/response DTOs and the same session state
 * table — multi-agent sessions appear in the dashboard alongside
 * single-agent sessions.
 *
 * To test side-by-side:
 *   POST /agent       → single-agent (PlannerService → AgentExecutorService)
 *   POST /agent/multi → multi-agent (SupervisorPlanner → SupervisorAgent)
 *
 * Same prompt, same docId, compare the step histories in the dashboard.
 */
@RestController
@RequestMapping("/agent/multi")
@RequiredArgsConstructor
public class MultiAgentController {

    private final SupervisorAgent supervisorAgent;

    @PostMapping
    public AgentResponse execute(@RequestBody AgentRequest request) {
        return supervisorAgent.handle(
                request.prompt(),
                request.docId(),
                request.topK()
        );
    }
}
