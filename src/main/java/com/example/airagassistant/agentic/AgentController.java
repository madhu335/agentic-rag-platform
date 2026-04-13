package com.example.airagassistant.agentic;

import com.example.airagassistant.agentic.dto.AgentRequest;
import com.example.airagassistant.agentic.dto.AgentResponse;
import com.example.airagassistant.agentic.dto.ContinueRequest;
import com.example.airagassistant.agentic.state.AgentSessionState;
import com.example.airagassistant.agentic.state.StepHistoryEntry;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/agent")
@RequiredArgsConstructor
public class AgentController {

    private final AgentExecutorService agentExecutorService;

    @PostMapping
    public AgentResponse run(@RequestBody AgentRequest request) {
        return agentExecutorService.execute(request);
    }

    @PostMapping("/continue")
    public AgentResponse continueSession(@RequestBody ContinueRequest request) {
        return agentExecutorService.continueExecution(request);
    }

    @GetMapping("/{sessionId}")
    public AgentSessionState getSession(@PathVariable String sessionId) {
        return agentExecutorService.getSessionState(sessionId);
    }

    @GetMapping("/{sessionId}/history")
    public List<StepHistoryEntry> getHistory(@PathVariable String sessionId) {
        AgentSessionState state = agentExecutorService.getSessionState(sessionId);
        return state.history() != null ? state.history() : List.of();
    }
}