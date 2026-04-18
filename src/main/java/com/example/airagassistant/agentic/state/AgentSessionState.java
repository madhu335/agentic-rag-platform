package com.example.airagassistant.agentic.state;

import com.example.airagassistant.judge.JudgeResult;

import java.time.Instant;
import java.util.List;

/**
 * Immutable session state for the agent executor.
 *
 * ── How to add a new field without breaking anything ──────────────────────
 *
 * 1. Add the field to the record component list below.
 * 2. Add a default value for it in the `empty()` factory (usually null / 0).
 * 3. Add a `withX()` copy method at the bottom of this file.
 * 4. Done. Every existing call site uses `StateBuilder` or `withX()` so they
 *    are unaffected by the new field.
 *
 * ── What NOT to do ─────────────────────────────────────────────────────────
 * Do NOT call `new AgentSessionState(...)` directly anywhere outside this
 * file. Use `AgentSessionState.empty(...)` for construction and `withX()`
 * for updates. That is the whole point of this pattern.
 */
public record AgentSessionState(
        String sessionId,
        String originalUserRequest,
        String currentUserRequest,
        String docId,

        ResearchSnapshot research,
        EmailSnapshot email,
        SmsSnapshot sms,
        VehicleSnapshot vehicle,
        ArticleSnapshot article,

        int researchAttempts,

        Instant createdAt,
        Instant updatedAt,
        List<StepHistoryEntry> history
) {

    // ─── Nested snapshot types ─────────────────────────────────────────────

    public record ResearchSnapshot(
            String summary,
            List<String> citations,
            Double retrievalScore,
            JudgeResult judge,
            List<String> chunks
    ) {}

    public record EmailSnapshot(
            String recipient,
            String subject,
            String body,
            EmailStatus status
    ) {}

    public record SmsSnapshot(
            String phoneNumber,
            String message,
            SmsStatus status
    ) {}

    public record VehicleSnapshot(
            String vehicleId,
            List<String> vehicleIds,
            String summary,
            String comparisonResult,
            List<String> specChunkIds,
            String rawSpecText,
            VehicleStepStatus status
    ) {}

    /**
     * Snapshot of the article agent's execution state.
     *
     * Visible at the top level of the session JSON alongside research/email/vehicle,
     * so the dashboard can show article-specific data without digging into history args.
     *
     * Fields:
     *   articleIds        — which articles were retrieved (from ArticleRagService)
     *   extractedVehicleIds — which vehicle IDs were extracted from article IDs
     *                         (the input to FetchVehicleSpecsTool)
     *   resolvedVehicleIds  — which of those actually returned spec chunks
     *                         (extractedVehicleIds minus the ones that returned empty)
     *   operation          — which execution path ran (ask_article, cross_article_search,
     *                         vehicle_scoped_search, vehicle_enriched_search)
     *   judgeScore         — final judge score (null if judge didn't run)
     *   chunkCount         — total context chunks fed to the LLM
     */
    public record ArticleSnapshot(
            List<String> articleIds,
            List<String> extractedVehicleIds,
            List<String> resolvedVehicleIds,
            String operation,
            Double judgeScore,
            int chunkCount
    ) {}

    // ─── Enums ─────────────────────────────────────────────────────────────

    public enum EmailStatus   { NOT_STARTED, DRAFTED, SENT, CANCELLED }
    public enum SmsStatus     { NOT_STARTED, COMPOSED, SENT, CANCELLED }
    public enum VehicleStepStatus { SPECS_FETCHED, SUMMARY_GENERATED, COMPARISON_DONE, ENRICHED }

    // ─── Factory ───────────────────────────────────────────────────────────

    /**
     * Create a brand-new session with all optional fields set to safe defaults.
     * Use this instead of calling the record constructor directly.
     *
     * Example:
     *   AgentSessionState state = AgentSessionState.empty(sessionId, prompt, docId);
     */
    public static AgentSessionState empty(String sessionId,
                                          String userRequest,
                                          String docId) {
        Instant now = Instant.now();
        return new AgentSessionState(
                sessionId,
                userRequest,
                userRequest,
                docId,
                null,   // research
                null,   // email
                null,   // sms
                null,   // vehicle
                null,   // article
                0,
                now,
                now,
                List.of()
        );
    }

    // ─── withX() copy helpers ──────────────────────────────────────────────
    // Each method returns a new record with one field replaced.
    // All other fields are copied from `this`.
    // Adding a new field to the record only requires adding one new withX() here.

    public AgentSessionState withResearch(ResearchSnapshot research) {
        return new AgentSessionState(sessionId, originalUserRequest, currentUserRequest, docId,
                research, email, sms, vehicle, article,
                researchAttempts, createdAt, Instant.now(), history);
    }

    public AgentSessionState withEmail(EmailSnapshot email) {
        return new AgentSessionState(sessionId, originalUserRequest, currentUserRequest, docId,
                research, email, sms, vehicle, article,
                researchAttempts, createdAt, Instant.now(), history);
    }

    public AgentSessionState withSms(SmsSnapshot sms) {
        return new AgentSessionState(sessionId, originalUserRequest, currentUserRequest, docId,
                research, email, sms, vehicle, article,
                researchAttempts, createdAt, Instant.now(), history);
    }

    public AgentSessionState withVehicle(VehicleSnapshot vehicle) {
        return new AgentSessionState(sessionId, originalUserRequest, currentUserRequest, docId,
                research, email, sms, vehicle, article,
                researchAttempts, createdAt, Instant.now(), history);
    }

    public AgentSessionState withArticle(ArticleSnapshot article) {
        return new AgentSessionState(sessionId, originalUserRequest, currentUserRequest, docId,
                research, email, sms, vehicle, article,
                researchAttempts, createdAt, Instant.now(), history);
    }

    public AgentSessionState withHistory(List<StepHistoryEntry> history) {
        return new AgentSessionState(sessionId, originalUserRequest, currentUserRequest, docId,
                research, email, sms, vehicle, article,
                researchAttempts, createdAt, Instant.now(), history);
    }

    public AgentSessionState withResearchAttempts(int researchAttempts) {
        return new AgentSessionState(sessionId, originalUserRequest, currentUserRequest, docId,
                research, email, sms, vehicle, article,
                researchAttempts, createdAt, Instant.now(), history);
    }

    public AgentSessionState withCurrentUserRequest(String currentUserRequest) {
        return new AgentSessionState(sessionId, originalUserRequest, currentUserRequest, docId,
                research, email, sms, vehicle, article,
                researchAttempts, createdAt, Instant.now(), history);
    }
}