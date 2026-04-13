package com.example.airagassistant.trace;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.context.Scope;
import org.springframework.stereotype.Component;

import java.util.Map;
import java.util.function.Supplier;

@Component
public class TraceHelper {

    private final Tracer tracer = GlobalOpenTelemetry.getTracer("ai-rag");

    public <T> T run(String spanName, Map<String, Object> attributes, Supplier<T> work) {
        Span span = tracer.spanBuilder(spanName).startSpan();
        try (Scope scope = span.makeCurrent()) {
            if (attributes != null) {
                attributes.forEach((k, v) -> setAttribute(span, k, v));
            }
            return work.get();
        } catch (Exception e) {
            span.recordException(e);
            throw e;
        } finally {
            span.end();
        }
    }

    public void runVoid(String spanName, Map<String, Object> attributes, Runnable work) {
        Span span = tracer.spanBuilder(spanName).startSpan();
        try (Scope scope = span.makeCurrent()) {
            if (attributes != null) {
                attributes.forEach((k, v) -> setAttribute(span, k, v));
            }
            work.run();
        } catch (Exception e) {
            span.recordException(e);
            throw e;
        } finally {
            span.end();
        }
    }
    public void addAttributes(Map<String, Object> attributes) {
        var span = io.opentelemetry.api.trace.Span.current();
        if (attributes == null || attributes.isEmpty()) {
            return;
        }

        attributes.forEach((k, v) -> setAttribute(span, k, v));
    }
    private void setAttribute(Span span, String key, Object value) {
        if (value == null) return;
        if (value instanceof String s) span.setAttribute(key, s);
        else if (value instanceof Integer i) span.setAttribute(key, i);
        else if (value instanceof Long l) span.setAttribute(key, l);
        else if (value instanceof Double d) span.setAttribute(key, d);
        else if (value instanceof Boolean b) span.setAttribute(key, b);
        else span.setAttribute(key, String.valueOf(value));
    }
}