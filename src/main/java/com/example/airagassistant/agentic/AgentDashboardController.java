package com.example.airagassistant.agentic;

import com.example.airagassistant.agentic.dto.AgentSessionSummaryDto;
import com.example.airagassistant.agentic.state.AgentSessionState;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/agent/dashboard")
@RequiredArgsConstructor
@CrossOrigin(origins = "*")
public class AgentDashboardController {

    private final AgentDashboardService dashboardService;

    @GetMapping("/sessions")
    public List<AgentSessionSummaryDto> recentSessions(
            @RequestParam(defaultValue = "25") int limit
    ) {
        return dashboardService.listRecentSessions(limit);
    }

    @GetMapping("/sessions/{sessionId}")
    public AgentSessionState getSession(@PathVariable String sessionId) {
        return dashboardService.getSession(sessionId);
    }
}