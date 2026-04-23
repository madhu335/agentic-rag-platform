package com.example.airagassistant.rag;

import com.example.airagassistant.domain.vehicle.VehicleRecord;
import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.Scope;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

@Service
public class ReRankService {

    private final Tracer tracer = GlobalOpenTelemetry.getTracer("ai-rag");
    private final TritonRerankerClient tritonRerankerClient;

    public ReRankService(TritonRerankerClient tritonRerankerClient) {
        this.tritonRerankerClient = tritonRerankerClient;
    }

    public List<SearchHit> rerank(String question, List<SearchHit> hits) {
        Span span = tracer.spanBuilder("rerank-score").startSpan();

        try (Scope scope = span.makeCurrent()) {
            span.setAttribute("langsmith.span.kind", "chain");
            span.setAttribute("gen_ai.prompt.0.role", "user");
            span.setAttribute("gen_ai.prompt.0.content", question == null ? "" : question);
            span.setAttribute("langsmith.metadata.input_count", hits == null ? 0 : hits.size());

            if (hits == null || hits.isEmpty()) {
                return List.of();
            }

            List<String> documents = hits.stream()
                    .map(hit -> hit.record().text())
                    .toList();

            List<Double> scores = tritonRerankerClient.score(question, documents);

            if (scores.size() != hits.size()) {
                throw new IllegalStateException(
                        "Reranker returned " + scores.size() + " scores for " + hits.size() + " hits"
                );
            }

            List<SearchHit> reranked = new ArrayList<>(hits.size());
            for (int i = 0; i < hits.size(); i++) {
                reranked.add(new SearchHit(hits.get(i).record(), scores.get(i)));
            }

            reranked.sort(Comparator.comparingDouble(SearchHit::score).reversed());

            span.setAttribute("langsmith.metadata.output_count", reranked.size());
            if (!reranked.isEmpty()) {
                span.setAttribute("langsmith.metadata.top_chunk_id", reranked.get(0).record().id());
                span.setAttribute("langsmith.metadata.top_score", reranked.get(0).score());
            }
            System.out.println("Reranker scores = " + scores);
            return reranked;
        } catch (Exception e) {
            span.recordException(e);
            throw e;
        } finally {
            span.end();
        }
    }

    private String buildDocument(VehicleRecord v) {
        StringBuilder sb = new StringBuilder();

        append(sb, "Vehicle ID", v.vehicleId());

        String title = joinNonBlank(
                safe(v.year() == 0 ? null : String.valueOf(v.year())),
                v.make(),
                v.model(),
                v.trim()
        );
        append(sb, "Title", title);

        append(sb, "Body Style", v.bodyStyle());
        append(sb, "Engine", v.engine());
        append(sb, "Horsepower", v.horsepower() == null ? null : String.valueOf(v.horsepower()));
        append(sb, "Torque", v.torque() == null ? null : String.valueOf(v.torque()));
        append(sb, "Drivetrain", v.drivetrain());
        append(sb, "Transmission", v.transmission());
        append(sb, "MPG City", v.mpgCity());
        append(sb, "MPG Highway", v.mpgHighway());
        append(sb, "MSRP", v.msrp());

        if (v.features() != null && !v.features().isEmpty()) {
            append(sb, "Features", String.join(", ", v.features()));
        }

        append(sb, "Summary", v.summary());

        return sb.toString().trim();
    }

    private void append(StringBuilder sb, String label, String value) {
        if (value != null && !value.isBlank()) {
            sb.append(label).append(": ").append(value).append('\n');
        }
    }

    private String joinNonBlank(String... values) {
        StringBuilder sb = new StringBuilder();
        for (String value : values) {
            if (value != null && !value.isBlank()) {
                if (sb.length() > 0) {
                    sb.append(' ');
                }
                sb.append(value);
            }
        }
        return sb.toString();
    }

    private String safe(String value) {
        return value == null ? "" : value;
    }
}